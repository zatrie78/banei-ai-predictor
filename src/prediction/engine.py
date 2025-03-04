def predict_race_outcome(race_info, model):
    """レース予測を行う関数"""
    # 入力されたレース情報から馬のデータを取り出す
    horses = race_info["horses"]
    
    # 馬のデータから特徴量（モデルが理解できる形式のデータ）を作る
    features_df = engineer_features(horses)
    
    # 学習済みモデルを使って予測を実行
    predictions = model.predict(features_df)
    
    # 予測結果を整理する
    ranked_horses = []
    for i, horse in enumerate(horses):
        # 予測順位
        predicted_rank = predictions[i]
        
        # 信頼度（どれくらい自信があるか）を計算
        # 信頼度は1位予測なら100%、最下位予測なら低くなるような計算にしています
        confidence = 100 * (1 - (predicted_rank - 1) / len(horses))
        
        # 結果を記録
        ranked_horses.append({
            "horse_name": horse["horse_name"],
            "jockey": horse["jockey"],
            "predicted_rank": predicted_rank,
            "confidence": confidence
        })
    
    # 予測順位で並べ替え
    ranked_horses.sort(key=lambda x: x["predicted_rank"])
    
    # 結果を返す
    return {
        "race_name": race_info["race_name"],
        "distance": race_info["distance"],
        "ranked_horses": ranked_horses
    }