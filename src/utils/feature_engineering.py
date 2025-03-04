import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def engineer_features(df, reference_data=None):
    """
    レースデータから特徴量を生成する関数
    
    Args:
        df (pd.DataFrame): 予測対象の馬のデータ
        reference_data (pd.DataFrame, optional): 過去のレースデータ（履歴から特徴量を作成する場合に使用）
    
    Returns:
        pd.DataFrame: エンジニアリングされた特徴量
    """
    # 結果のコピーを作成
    result_df = df.copy()
    
    # 基本的な特徴量変換
    
    # 1. 体重変化率の計算
    if 'weight_change' in result_df.columns and 'horse_weight' in result_df.columns:
        result_df['weight_change_rate'] = result_df['weight_change'] / result_df['horse_weight'] * 100
    else:
        result_df['weight_change_rate'] = 0
    
    # 参照データが提供されている場合は、履歴ベースの特徴量を作成
    if reference_data is not None:
        result_df = add_historical_features(result_df, reference_data)
    else:
        # 参照データがない場合は、基本的な推定値を使用
        result_df = add_estimated_features(result_df)
    
    # 必要な特徴量のみを選択
    required_features = [
        'horse_id', 'jockey_id', 'horse_weight', 'weight_change_rate',
        'last_5_races_avg', 'rides_together', 'avg_rank_together',
        'win_rate_0_2', 'win_rate_2_5', 'win_rate_5_10', 'win_rate_10_20',
        'roi_0_2', 'roi_2_5', 'roi_5_10', 'roi_10_20'
    ]
    
    # 存在する特徴量のみを選択（すべての必須特徴量がなくてもエラーにならないように）
    available_features = [f for f in required_features if f in result_df.columns]
    features_df = result_df[available_features]
    
    # 欠損値の処理
    features_df = handle_missing_values(features_df, required_features)
    
    return features_df

def add_historical_features(df, reference_data):
    """
    過去のレースデータから履歴ベースの特徴量を追加する
    
    Args:
        df (pd.DataFrame): 予測対象の馬のデータ
        reference_data (pd.DataFrame): 過去のレースデータ
    
    Returns:
        pd.DataFrame: 履歴特徴量が追加されたデータフレーム
    """
    # 日付でソート
    reference_data = reference_data.sort_values('race_date')
    
    # IDマッピングの確認
    if 'horse_name' in df.columns and 'horse_id' not in df.columns:
        # horse_id のマッピングを作成
        horse_mapping = {}
        if 'horse_name' in reference_data.columns and 'horse_id' in reference_data.columns:
            horse_mapping = dict(zip(reference_data['horse_name'], reference_data['horse_id']))
        
        # マッピングを適用
        df['horse_id'] = df['horse_name'].map(horse_mapping)
        
        # 新しい馬に対してはIDを割り当て
        unknown_horses = df[df['horse_id'].isna()]['horse_name'].unique()
        max_id = reference_data['horse_id'].max() if 'horse_id' in reference_data.columns else 0
        for i, horse in enumerate(unknown_horses):
            df.loc[df['horse_name'] == horse, 'horse_id'] = max_id + i + 1
    
    # 同様に騎手IDを処理
    if 'jockey' in df.columns and 'jockey_id' not in df.columns:
        jockey_mapping = {}
        if 'jockey' in reference_data.columns and 'jockey_id' in reference_data.columns:
            jockey_mapping = dict(zip(reference_data['jockey'], reference_data['jockey_id']))
        
        df['jockey_id'] = df['jockey'].map(jockey_mapping)
        
        unknown_jockeys = df[df['jockey_id'].isna()]['jockey'].unique()
        max_id = reference_data['jockey_id'].max() if 'jockey_id' in reference_data.columns else 0
        for i, jockey in enumerate(unknown_jockeys):
            df.loc[df['jockey'] == jockey, 'jockey_id'] = max_id + i + 1
    
    # 1. 馬の過去成績
    if 'horse_id' in df.columns and 'horse_id' in reference_data.columns and 'rank' in reference_data.columns:
        # 直近5走の平均順位
        last_5_races = {}
        for horse_id in df['horse_id'].unique():
            horse_races = reference_data[reference_data['horse_id'] == horse_id]
            if len(horse_races) > 0:
                last_5 = horse_races.tail(5)['rank'].mean()
                last_5_races[horse_id] = last_5
        
        df['last_5_races_avg'] = df['horse_id'].map(last_5_races)
    
    # 2. 騎手の成績
    if 'jockey_id' in df.columns and 'jockey_id' in reference_data.columns:
        # オッズ帯ごとの勝率と回収率
        odds_ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, float('inf'))]
        
        for min_odds, max_odds in odds_ranges:
            win_rates = {}
            roi_values = {}
            
            for jockey_id in df['jockey_id'].unique():
                jockey_races = reference_data[reference_data['jockey_id'] == jockey_id]
                
                if 'odds' in jockey_races.columns and 'rank' in jockey_races.columns:
                    # 指定オッズ帯のレース
                    odds_range_races = jockey_races[(jockey_races['odds'] >= min_odds) & 
                                                    (jockey_races['odds'] < max_odds)]
                    
                    if len(odds_range_races) > 0:
                        # 勝率の計算
                        win_count = len(odds_range_races[odds_range_races['rank'] == 1])
                        win_rate = win_count / len(odds_range_races)
                        win_rates[jockey_id] = win_rate
                        
                        # 回収率の計算
                        roi = (odds_range_races[odds_range_races['rank'] == 1]['odds'].sum() / 
                               len(odds_range_races)) - 1
                        roi_values[jockey_id] = roi
            
            df[f'win_rate_{min_odds}_{max_odds}'] = df['jockey_id'].map(win_rates)
            df[f'roi_{min_odds}_{max_odds}'] = df['jockey_id'].map(roi_values)
    
    # 3. 馬と騎手の組み合わせの特徴
    if ('horse_id' in df.columns and 'jockey_id' in df.columns and 
        'horse_id' in reference_data.columns and 'jockey_id' in reference_data.columns):
        
        rides_together = {}
        avg_rank_together = {}
        
        for _, row in df.iterrows():
            horse_jockey_key = (row['horse_id'], row['jockey_id'])
            
            # この馬と騎手の組み合わせの過去のレース
            combo_races = reference_data[(reference_data['horse_id'] == row['horse_id']) & 
                                         (reference_data['jockey_id'] == row['jockey_id'])]
            
            if len(combo_races) > 0:
                rides_together[horse_jockey_key] = len(combo_races)
                avg_rank_together[horse_jockey_key] = combo_races['rank'].mean()
        
        # DataFrameに追加
        for idx, row in df.iterrows():
            key = (row['horse_id'], row['jockey_id'])
            df.at[idx, 'rides_together'] = rides_together.get(key, 0)
            df.at[idx, 'avg_rank_together'] = avg_rank_together.get(key, 0)
    
    return df

def add_estimated_features(df):
    """
    参照データがない場合、基本的な推定値を使って特徴量を作成する
    
    Args:
        df (pd.DataFrame): 予測対象の馬のデータ
    
    Returns:
        pd.DataFrame: 推定特徴量が追加されたデータフレーム
    """
    n_rows = len(df)
    
    # 馬IDと騎手IDの作成（もしなければ）
    if 'horse_name' in df.columns and 'horse_id' not in df.columns:
        horse_ids = {name: i+1 for i, name in enumerate(df['horse_name'].unique())}
        df['horse_id'] = df['horse_name'].map(horse_ids)
    
    if 'jockey' in df.columns and 'jockey_id' not in df.columns:
        jockey_ids = {name: i+1 for i, name in enumerate(df['jockey'].unique())}
        df['jockey_id'] = df['jockey'].map(jockey_ids)
    
    # 直近5走の平均順位（標準的な値で代用）
    df['last_5_races_avg'] = 4.5
    
    # 騎手の過去成績（標準的な値で代用）
    for min_odds, max_odds in [(0, 2), (2, 5), (5, 10), (10, 20), (20, float('inf'))]:
        # 勝率はオッズ帯によって変化させる（低オッズほど高勝率）
        base_win_rate = 0.3 * (1 / (min_odds + 1))
        df[f'win_rate_{min_odds}_{max_odds}'] = base_win_rate
        
        # 回収率も同様に設定
        base_roi = -0.1 + (0.05 * min_odds)  # 基本的には負の回収率
        df[f'roi_{min_odds}_{max_odds}'] = base_roi
    
    # 馬と騎手の組み合わせ（標準的な値で代用）
    df['rides_together'] = 2  # 平均的に数回の組み合わせ
    df['avg_rank_together'] = 4.0  # 平均的な着順
    
    return df

def handle_missing_values(df, required_features):
    """
    必要な特徴量の欠損値を処理する
    
    Args:
        df (pd.DataFrame): 特徴量データフレーム
        required_features (list): 必要な特徴量のリスト
    
    Returns:
        pd.DataFrame: 欠損値が処理されたデータフレーム
    """
    # 欠損している特徴量を特定
    missing_features = [f for f in required_features if f not in df.columns]
    
    # 欠損している特徴量に適切なデフォルト値を設定
    for feature in missing_features:
        if feature == 'horse_id':
            df[feature] = range(1, len(df) + 1)
        elif feature == 'jockey_id':
            df[feature] = 1  # すべて同じ騎手と仮定
        elif feature == 'horse_weight':
            df[feature] = 800  # 平均的な馬体重
        elif feature == 'weight_change_rate':
            df[feature] = 0  # 変化なしと仮定
        elif feature == 'last_5_races_avg':
            df[feature] = 4.5  # 平均的な着順
        elif feature == 'rides_together':
            df[feature] = 2  # 平均的な回数
        elif feature == 'avg_rank_together':
            df[feature] = 4.0  # 平均的な着順
        elif feature.startswith('win_rate'):
            # オッズ帯に応じた勝率
            odds_range = feature.split('_')[2:]
            min_odds = float(odds_range[0])
            base_win_rate = 0.3 * (1 / (min_odds + 1))
            df[feature] = base_win_rate
        elif feature.startswith('roi'):
            # オッズ帯に応じた回収率
            odds_range = feature.split('_')[1:]
            min_odds = float(odds_range[0])
            base_roi = -0.1 + (0.05 * min_odds)
            df[feature] = base_roi
    
    # 既存の列の欠損値を処理
    for feature in df.columns:
        if df[feature].isna().any():
            if feature in ['horse_id', 'jockey_id']:
                # IDの欠損値は新しい値を割り当て
                max_id = df[feature].max()
                df[feature] = df[feature].fillna(max_id + 1)
            elif feature == 'horse_weight':
                df[feature] = df[feature].fillna(800)
            elif feature == 'weight_change_rate':
                df[feature] = df[feature].fillna(0)
            elif feature in ['last_5_races_avg', 'avg_rank_together']:
                df[feature] = df[feature].fillna(4.5)
            elif feature == 'rides_together':
                df[feature] = df[feature].fillna(2)
            elif feature.startswith('win_rate'):
                # オッズ帯に応じた標準的な勝率
                odds_range = feature.split('_')[2:]
                min_odds = float(odds_range[0]) if len(odds_range) > 0 else 5
                base_win_rate = 0.3 * (1 / (min_odds + 1))
                df[feature] = df[feature].fillna(base_win_rate)
            elif feature.startswith('roi'):
                # オッズ帯に応じた標準的な回収率
                odds_range = feature.split('_')[1:]
                min_odds = float(odds_range[0]) if len(odds_range) > 0 else 5
                base_roi = -0.1 + (0.05 * min_odds)
                df[feature] = df[feature].fillna(base_roi)
    
    return df

def normalize_features(df):
    """
    特徴量を正規化する（必要に応じて使用）
    
    Args:
        df (pd.DataFrame): 特徴量データフレーム
    
    Returns:
        pd.DataFrame: 正規化された特徴量データフレーム
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    exclude_cols = ['horse_id', 'jockey_id']
    
    for col in numeric_features:
        if col not in exclude_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
    
    return df

def create_jockey_features(df):
    """
    騎手に関する特徴を作成
    
    Args:
        df (pd.DataFrame): レースデータ
    
    Returns:
        pd.DataFrame: 騎手特徴が追加されたデータフレーム
    """
    # 日付でソート
    if 'race_date' in df.columns:
        df = df.sort_values('race_date')
    
    # 騎手ごとの集計を作成
    jockey_stats = []
    
    for jockey_id in df['jockey_id'].unique():
        jockey_df = df[df['jockey_id'] == jockey_id]
        
        # オッズ帯ごとの成績を集計
        odds_ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]
        odds_stats = {}
        
        for odds_min, odds_max in odds_ranges:
            odds_range_df = jockey_df[
                (jockey_df['odds'] >= odds_min) & 
                (jockey_df['odds'] < odds_max)
            ]
            if len(odds_range_df) > 0:
                win_rate = len(odds_range_df[odds_range_df['rank'] == 1]) / len(odds_range_df)
                roi = (odds_range_df[odds_range_df['rank'] == 1]['odds'].sum() / len(odds_range_df)) - 1
                
                odds_stats[f'win_rate_{odds_min}_{odds_max}'] = win_rate
                odds_stats[f'roi_{odds_min}_{odds_max}'] = roi
        
        jockey_stats.append({
            'jockey_id': jockey_id,
            'jockey_name': jockey_df['jockey'].iloc[0] if 'jockey' in jockey_df.columns else None,
            **odds_stats
        })
    
    jockey_stats_df = pd.DataFrame(jockey_stats)
    return jockey_stats_df

def create_horse_features(df):
    """
    馬に関する特徴を作成
    
    Args:
        df (pd.DataFrame): レースデータ
    
    Returns:
        pd.DataFrame: 馬特徴が追加されたデータフレーム
    """
    # 日付でソート
    if 'race_date' in df.columns:
        df = df.sort_values(['horse_id', 'race_date'])
    
    # 前走の情報を追加
    df['prev_rank'] = df.groupby('horse_id')['rank'].shift(1)
    if 'time_seconds' in df.columns:
        df['prev_time'] = df.groupby('horse_id')['time_seconds'].shift(1)
    if 'horse_weight' in df.columns:
        df['prev_weight'] = df.groupby('horse_id')['horse_weight'].shift(1)
    
    # 馬体重の変化率
    if 'horse_weight' in df.columns and 'prev_weight' in df.columns:
        df['weight_change_rate'] = (df['horse_weight'] - df['prev_weight']) / df['prev_weight'] * 100
    
    # 直近5走の成績
    df['last_5_races_avg'] = df.groupby('horse_id')['rank'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    return df

def create_horse_jockey_features(df):
    """
    騎手と馬の組み合わせに関する特徴を作成
    
    Args:
        df (pd.DataFrame): レースデータ
    
    Returns:
        pd.DataFrame: 馬と騎手の組み合わせ特徴が追加されたデータフレーム
    """
    # 騎手と馬の組み合わせごとの成績を集計
    combinations = df.groupby(['horse_id', 'jockey_id']).agg({
        'rank': ['count', 'mean', 'min'],
        'time_seconds': 'mean' if 'time_seconds' in df.columns else 'count'
    }).reset_index()
    
    combinations.columns = [
        'horse_id', 'jockey_id', 
        'rides_together', 'avg_rank_together', 'best_rank_together',
        'avg_time_together' if 'time_seconds' in df.columns else 'rides_count'
    ]
    
    # 元のデータフレームとマージ
    df = df.merge(
        combinations,
        on=['horse_id', 'jockey_id'],
        how='left'
    )
    
    return df