"""Hidden Markov Model for regime detection."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')


class RegimeHMM:
    def __init__(self, n_states=3, cov_type='full', random_state=42, n_iter=2000):
        self.n_states = n_states
        self.cov_type = cov_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_states, covariance_type=cov_type,
            n_iter=n_iter, random_state=random_state, tol=1e-4)
        self.fitted = False
        self.feature_cols = None

    def fit(self, features_df, feature_cols):
        self.feature_cols = feature_cols
        sub = features_df[feature_cols].dropna()
        X = sub.values
        self.train_index = sub.index
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.fitted = True
        # Compute state means in original scale
        states = self.model.predict(Xs)
        self.state_means_ = {}
        for s in range(self.n_states):
            idx = (states == s)
            if np.sum(idx) > 0:
                self.state_means_[s] = dict(
                    zip(feature_cols, np.mean(X[idx], axis=0)))
            else:
                self.state_means_[s] = {c: np.nan for c in feature_cols}
        self.train_states = pd.Series(states, index=self.train_index, name='state')
        return self

    def predict_states(self, features_df):
        X = features_df[self.feature_cols].values
        mask = ~np.any(np.isnan(X), axis=1)
        result = pd.Series(index=features_df.index, data=np.nan, dtype=float)
        if mask.sum() == 0:
            return result
        Xs = self.scaler.transform(X[mask])
        states = self.model.predict(Xs)
        result.iloc[np.where(mask)] = states.astype(float)
        return result

    def predict_proba(self, features_df):
        X = features_df[self.feature_cols].values
        mask = ~np.any(np.isnan(X), axis=1)
        proba = np.full((len(features_df), self.n_states), np.nan)
        if mask.sum() == 0:
            return pd.DataFrame(proba, index=features_df.index)
        Xs = self.scaler.transform(X[mask])
        proba[mask] = self.model.predict_proba(Xs)
        return pd.DataFrame(proba, index=features_df.index,
                            columns=[f'state_{i}' for i in range(self.n_states)])

    def score(self, features_df):
        sub = features_df[self.feature_cols].dropna()
        Xs = self.scaler.transform(sub.values)
        return self.model.score(Xs)

    def get_transition_matrix(self):
        return self.model.transmat_

    def get_aic_bic(self, features_df):
        sub = features_df[self.feature_cols].dropna()
        Xs = self.scaler.transform(sub.values)
        ll = self.model.score(Xs)  # total log-likelihood
        n_features = len(self.feature_cols)
        n_params = (self.n_states ** 2 - self.n_states  # transition
                    + self.n_states * n_features  # means
                    + self.n_states * n_features * (n_features + 1) // 2)  # covs
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(len(Xs))
        return aic, bic


def map_states_to_regimes(hmm_model, features_df,
                          lyap_col='lyap', hurst_col='hurst',
                          perm_col='perm_entropy'):
    """Map HMM states to regime labels based on feature means."""
    states = hmm_model.predict_states(features_df)
    state_stats = {}
    for s in range(hmm_model.n_states):
        idx = (states == s)
        if idx.sum() == 0:
            state_stats[s] = None
            continue
        state_stats[s] = features_df.loc[idx, [lyap_col, hurst_col, perm_col, 'vol_short', 'ret']].mean()
    # Chaotic: highest lyapunov exponent
    lyap_means = {s: (state_stats[s][lyap_col] if state_stats[s] is not None else -np.inf)
                  for s in range(hmm_model.n_states)}
    s_chaos = max(lyap_means, key=lyap_means.get)
    mapping = {s_chaos: 'chaotic'}
    remaining = [s for s in range(hmm_model.n_states) if s != s_chaos]
    if len(remaining) == 2:
        hurst_means = {s: (state_stats[s][hurst_col] if state_stats[s] is not None else 0.5)
                       for s in remaining}
        s_trend = max(hurst_means, key=hurst_means.get)
        s_mr = min(hurst_means, key=hurst_means.get)
        mapping[s_trend] = 'trending'
        mapping[s_mr] = 'mean_reverting'
    elif len(remaining) == 1:
        mapping[remaining[0]] = 'trending'
    return mapping, state_stats
