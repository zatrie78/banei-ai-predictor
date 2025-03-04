import requests
import os
import json
import logging
from pathlib import Path

class AnthropicClient:
    """Anthropic APIとの連携を行うクラス"""
    
    def __init__(self, config_path="config/settings.json"):
        """初期化"""
        self.config_path = config_path
        self.api_key = self.get_api_key()
        self.setup_logging()
        
    def setup_logging(self):
        """ログ設定"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'anthropic.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_api_key(self):
        """API keyを取得する"""
        # 設定ファイルから読み込む
        config_path = Path(self.config_path)
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    return settings.get("api_key", "")
            except Exception as e:
                print(f"設定ファイルの読み込みエラー: {e}")
                return ""
        
        # 環境変数をフォールバックとして使用
        return os.environ.get("ANTHROPIC_API_KEY", "")
    
    def save_api_key(self, api_key):
        """API keyを保存する"""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(exist_ok=True)
        
        settings = {}
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception as e:
                self.logger.error(f"設定ファイルの読み込みエラー: {e}")
        
        settings["api_key"] = api_key
        
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            self.api_key = api_key
            self.logger.info("APIキーが保存されました")
            return True
        except Exception as e:
            self.logger.error(f"設定ファイルの保存エラー: {e}")
            return False
    
    def call_api(self, prompt, max_tokens=1000, model="claude-3-7-sonnet-latest"):
        """Anthropic APIを呼び出す関数"""
        if not self.api_key:
            self.logger.warning("APIキーが設定されていません")
            return "APIキーが設定されていません。設定タブからAPIキーを設定してください。"
        
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        self.logger.info(f"APIリクエスト送信: model={model}, max_tokens={max_tokens}")
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30  # タイムアウト設定
            )
            
            if response.status_code == 200:
                self.logger.info("APIリクエスト成功")
                return response.json()["content"][0]["text"]
            else:
                error_msg = f"APIエラー: {response.status_code} {response.text}"
                self.logger.error(error_msg)
                
                # 特定のエラーに対するユーザーフレンドリーなメッセージ
                if response.status_code == 401:
                    return "APIキーが無効です。正しいAPIキーを設定してください。"
                elif response.status_code == 429:
                    return "APIリクエストの制限に達しました。しばらく待ってから再試行してください。"
                else:
                    return error_msg
                    
        except requests.exceptions.Timeout:
            error_msg = "APIリクエストがタイムアウトしました。ネットワーク接続を確認してください。"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"API呼び出しエラー: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def test_connection(self):
        """APIの接続テスト"""
        test_prompt = "こんにちは。これは接続テストです。1+1はいくつですか？"
        try:
            response = self.call_api(test_prompt, max_tokens=50)
            if "APIエラー" in response or "API呼び出しエラー" in response:
                return False, response
            return True, "API接続テスト成功"
        except Exception as e:
            return False, f"API接続テストエラー: {str(e)}"
    
    def generate_prediction_explanation(self, race_info, predictions, detail_level="standard"):
        """予測結果の説明を生成する関数
        
        Args:
            race_info (dict): レース情報
            predictions (dict): 予測結果
            detail_level (str): 解説の詳細度 ("brief", "standard", "detailed")
        
        Returns:
            str: 生成された解説文
        """
        # 上位馬の情報をフォーマット
        horses_to_show = 3 if detail_level != "detailed" else 5
        horses_info = "\n".join([
            f"- {i+1}着予想: {horse['horse_name']} (騎手: {horse['jockey']}, 信頼度: {horse['confidence']:.1f}%)"
            for i, horse in enumerate(predictions["ranked_horses"][:horses_to_show])
        ])
        
        # 詳細度に応じた文字数とフォーマットの調整
        if detail_level == "brief":
            word_count = "150-200文字"
            format_instruction = "簡潔に要点のみ"
        elif detail_level == "standard":
            word_count = "300-400文字"
            format_instruction = "バランスよく"
        else:  # detailed
            word_count = "500-600文字"
            format_instruction = "詳細に、各馬の特徴や考察を含めて"
        
        # 馬場状態などの追加情報を含める
        additional_info = ""
        if "track_condition" in race_info:
            additional_info += f"- 馬場状態: {race_info['track_condition']}\n"
        if "weather" in race_info:
            additional_info += f"- 天候: {race_info['weather']}\n"
        
        prompt = f"""あなたはばんえい競馬の専門家です。以下の情報に基づいて、このレースの予想解説を{word_count}程度で{format_instruction}提供してください。

レース情報:
- レース名: {race_info['race_name']}
- 距離: {race_info['distance']}m
{additional_info}

AI予想:
{horses_info}

本レースの展開予想、注目すべき馬の特徴、勝負どころについて解説してください。
また、予想の根拠や考慮すべき要素についても言及してください。
レース特有の状況（馬場状態、馬の調子、騎手の相性など）を考慮した具体的な解説を心がけてください。"""
        
        self.logger.info(f"予測解説生成リクエスト: レース名={race_info['race_name']}, 詳細度={detail_level}")
        
        # 詳細度に応じたトークン数の調整
        max_tokens = 800 if detail_level == "brief" else (1000 if detail_level == "standard" else 1500)
        
        return self.call_api(prompt, max_tokens=max_tokens)

# 単体で実行する場合のサンプルコード
if __name__ == "__main__":
    client = AnthropicClient()
    
    # 接続テスト
    success, message = client.test_connection()
    print(f"接続テスト結果: {'成功' if success else '失敗'} - {message}")
    
    if success:
        # サンプル予測結果
        sample_race_info = {
            "race_name": "第32回 ばんえい大賞典",
            "distance": 200,
            "track_condition": "重",
            "weather": "曇り"
        }
        
        sample_predictions = {
            "ranked_horses": [
                {"horse_name": "ホクショウマサル", "jockey": "鈴木恵介", "confidence": 85.5},
                {"horse_name": "カネタマル", "jockey": "藤本彰人", "confidence": 72.3},
                {"horse_name": "ホクトセンショー", "jockey": "阿部武瑠", "confidence": 65.8},
                {"horse_name": "ホクショウタケ", "jockey": "島津新", "confidence": 58.4},
                {"horse_name": "フジノオリオン", "jockey": "松本秀之", "confidence": 52.1}
            ]
        }
        
        # 解説生成テスト
        explanation = client.generate_prediction_explanation(sample_race_info, sample_predictions)
        print("\n=== 生成された解説 ===")
        print(explanation)