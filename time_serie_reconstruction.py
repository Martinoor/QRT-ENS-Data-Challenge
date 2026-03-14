import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from sklearn import linear_model

import lightgbm as lgbm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from numba import jit, prange
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Load data

X_train = pd.read_csv('data/X_train.csv',index_col='ROW_ID')
X_test = pd.read_csv('data/X_test.csv',index_col='ROW_ID')

y_train = pd.read_csv('data/y_train.csv',index_col='ROW_ID')
sample_submission = pd.read_csv('data/sample_submission.csv',index_col='ROW_ID')


#reconstruct the time series

# ============================================
# VERSION OPTIMISÉE ULTRA-RAPIDE
# ============================================

@jit(nopython=True, parallel=True)
def compute_overlap_score_fast(arr_i, arr_j):
    """
    Compare RET_19...RET_1 de i avec RET_20...RET_2 de j
    Compilé avec Numba pour vitesse maximale
    """
    # arr_i a 19 éléments (RET_19 à RET_1)
    # arr_j a 19 éléments (RET_20 à RET_2)
    score = 0.0
    for k in prange(19):
        diff = abs(arr_i[k] - arr_j[k])
        score += diff
    return -score  # Négatif car on veut minimiser la distance


def find_best_successor(current_row, candidates_df, ret_cols_shifted, vol_cols_shifted, 
                        ret_cols_current, vol_cols_current, used_indices):
    """
    Trouve le meilleur successeur pour current_row parmi les candidats
    Optimisé avec vectorisation
    """
    if len(candidates_df) == 0:
        return None
    
    # Extraire les valeurs pour comparaison vectorisée
    current_ret = current_row[ret_cols_shifted].values
    current_vol = current_row[vol_cols_shifted].values
    
    # Filtrer les candidats non utilisés
    candidates = candidates_df[~candidates_df.index.isin(used_indices)]
    
    if len(candidates) == 0:
        return None
    
    # Comparaison vectorisée (beaucoup plus rapide)
    candidates_ret = candidates[ret_cols_current].values
    candidates_vol = candidates[vol_cols_current].values
    
    # Distance L1 (Manhattan) - plus rapide que L2
    ret_distances = np.abs(candidates_ret - current_ret).sum(axis=1)
    vol_distances = np.abs(candidates_vol - current_vol).sum(axis=1)
    
    # Score combiné (pondérer returns plus que volumes)
    total_distances = ret_distances + 0.3 * vol_distances
    
    # Trouver le meilleur
    best_idx = total_distances.argmin()
    best_candidate_idx = candidates.index[best_idx]
    best_score = -total_distances[best_idx]
    
    return best_candidate_idx, best_score


def reconstruct_allocation_fast(alloc_df):
    """
    Reconstruction optimisée pour UNE allocation
    Approche greedy avec heuristiques
    """
    n = len(alloc_df)
    
    if n <= 1:
        alloc_df['reconstructed_order'] = 0
        alloc_df['confidence_score'] = 1.0
        return alloc_df
    
    # Colonnes pour la comparaison
    ret_cols_shifted = [f'RET_{i}' for i in range(19, 0, -1)]  # RET_19 à RET_1
    ret_cols_current = [f'RET_{i}' for i in range(20, 1, -1)]  # RET_20 à RET_2
    
    vol_cols_shifted = [f'SIGNED_VOLUME_{i}' for i in range(19, 0, -1)]
    vol_cols_current = [f'SIGNED_VOLUME_{i}' for i in range(20, 1, -1)]
    
    # Trouver le point de départ (heuristique: observation avec RET_20 le plus ancien)
    # On peut utiliser la somme des returns absolus comme proxy
    start_scores = alloc_df[[f'RET_{i}' for i in range(20, 10, -1)]].abs().sum(axis=1)
    start_idx = start_scores.idxmax()  # Celui avec le plus de variation = plus ancien
    
    # Construction de la séquence
    sequence = [start_idx]
    used_indices = {start_idx}
    confidence_scores = [1.0]
    
    current_idx = start_idx
    
    for step in range(n - 1):
        current_row = alloc_df.loc[current_idx]
        
        result = find_best_successor(
            current_row, alloc_df, 
            ret_cols_shifted, vol_cols_shifted,
            ret_cols_current, vol_cols_current,
            used_indices
        )
        
        if result is None:
            # Pas de successeur trouvé, prendre le premier non utilisé
            remaining = set(alloc_df.index) - used_indices
            if remaining:
                next_idx = list(remaining)[0]
                confidence_scores.append(0.0)
            else:
                break
        else:
            next_idx, score = result
            # Normaliser le score pour avoir une "confiance"
            confidence = max(0, min(1, 1 / (1 + abs(score) / 10)))
            confidence_scores.append(confidence)
        
        sequence.append(next_idx)
        used_indices.add(next_idx)
        current_idx = next_idx
    
    # Créer le résultat
    result_df = alloc_df.loc[sequence].copy()
    result_df['reconstructed_order'] = range(len(sequence))
    result_df['confidence_score'] = confidence_scores
    
    return result_df


def reconstruct_all_allocations_parallel(df, n_jobs=-1):
    """
    Reconstruit toutes les allocations en parallèle
    """
    from joblib import Parallel, delayed
    
    allocations = df['ALLOCATION'].unique()
    print(f"📊 Reconstruction de {len(allocations)} allocations...")
    print(f"💾 {len(df)} observations au total")
    
    # Fonction wrapper pour le parallélisme
    def process_allocation(alloc):
        alloc_data = df[df['ALLOCATION'] == alloc].copy()
        return reconstruct_allocation_fast(alloc_data)
    
    # Traitement en parallèle avec barre de progression
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_allocation)(alloc) 
        for alloc in tqdm(allocations, desc="Reconstruction")
    )
    
    # Combiner les résultats
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df


def validate_reconstruction(df):
    """
    Validation de la qualité de la reconstruction
    """
    print("\n🔍 VALIDATION DE LA RECONSTRUCTION")
    print("=" * 60)
    
    validations = []
    
    for alloc in df['ALLOCATION'].unique():
        alloc_data = df[df['ALLOCATION'] == alloc].sort_values('reconstructed_order')
        
        if len(alloc_data) < 2:
            continue
        
        errors = 0
        total_checks = 0
        
        for i in range(len(alloc_data) - 1):
            current = alloc_data.iloc[i]
            next_obs = alloc_data.iloc[i + 1]
            
            # Vérification: target[i] devrait être proche de RET_1[i+1]
            if 'target' in current.index and 'RET_1' in next_obs.index:
                diff = abs(current['target'] - next_obs['RET_1'])
                if diff > 0.001:  # Seuil de tolérance
                    errors += 1
                total_checks += 1
        
        if total_checks > 0:
            accuracy = 1 - (errors / total_checks)
            validations.append({
                'allocation': alloc,
                'n_obs': len(alloc_data),
                'accuracy': accuracy,
                'avg_confidence': alloc_data['confidence_score'].mean()
            })
    
    validation_df = pd.DataFrame(validations)
    
    print(f"\n✅ Accuracy moyenne: {validation_df['accuracy'].mean():.2%}")
    print(f"📊 Confiance moyenne: {validation_df['avg_confidence'].mean():.2%}")
    print(f"\n🔝 Top 5 meilleures reconstructions:")
    print(validation_df.nlargest(5, 'accuracy')[['allocation', 'n_obs', 'accuracy', 'avg_confidence']])
    print(f"\n⚠️  Top 5 pires reconstructions:")
    print(validation_df.nsmallest(5, 'accuracy')[['allocation', 'n_obs', 'accuracy', 'avg_confidence']])
    
    return validation_df


# ============================================
# USAGE PRINCIPAL
# ============================================

def main_reconstruction_pipeline(X_train_path, y_train_path=None):
    """
    Pipeline complet de reconstruction
    """
    print("🚀 DÉBUT DE LA RECONSTRUCTION TEMPORELLE")
    print("=" * 60)
    
    # Chargement
    print("📂 Chargement des données...")
    X_train = pd.read_csv(X_train_path)
    
    if y_train_path:
        y_train = pd.read_csv(y_train_path)
        df = X_train.merge(y_train, on='ROW_ID', how='left')
    else:
        df = X_train.copy()
    
    print(f"✓ {len(df)} lignes chargées")
    print(f"✓ {df['ALLOCATION'].nunique()} allocations uniques")
    
    # Reconstruction
    import time
    start_time = time.time()
    
    reconstructed_df = reconstruct_all_allocations_parallel(df, n_jobs=-1)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Temps d'exécution: {elapsed/60:.2f} minutes")
    
    # Validation si on a le target
    if 'target' in reconstructed_df.columns:
        validation_results = validate_reconstruction(reconstructed_df)
    
    # Sauvegarder
    output_path = 'X_train_reconstructed.csv'
    reconstructed_df.to_csv(output_path, index=False)
    print(f"\n💾 Résultats sauvegardés: {output_path}")
    
    return reconstructed_df


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Utilisation
    reconstructed = main_reconstruction_pipeline(
        X_train_path='data/X_train.csv',
        y_train_path='data/y_train.csv'
    )
    
    print("\n✨ RECONSTRUCTION TERMINÉE!")
    print(f"📊 Nouvelles colonnes ajoutées:")
    print("   - reconstructed_order: Position dans la séquence temporelle")
    print("   - confidence_score: Confiance de la reconstruction (0-1)")
    
