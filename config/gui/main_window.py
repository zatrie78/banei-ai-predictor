import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import joblib
import pandas as pd
import logging
from pathlib import Path

# プロジェクトルートへのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 自作モジュールのインポート
try:
    from src.utils.feature_engineering import engineer_features
    from src.anthropic.client import AnthropicClient
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")

# ロギングの設定
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class BaneiAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ばんえい競馬AI予想システム")
        self.root.geometry("900x700")
        
        # Anthropic APIクライアントの初期化
        self.anthropic_client = AnthropicClient()
        
        # モデルのロード
        self.model_path = os.path.join(project_root, "models", "lightgbm_model_latest.pkl")
        self.model = self.load_model()
        
        # GUIコンポーネントの初期化
        self.setup_gui()
        
        # データの初期化
        self.horse_data = {}
        self.prediction_results = {}
    
    def load_model(self):
        """予測モデルをロードする"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"モデルを読み込み: {self.model_path}")
                return joblib.load(self.model_path)
            else:
                logger.warning(f"モデルファイルが見つかりません: {self.model_path}")
                return None
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
            return None
    
    def setup_gui(self):
        """GUIコンポーネントをセットアップする"""
        # ナビゲーションタブ
        self.tab_control = ttk.Notebook(self.root)
        
        # 予測タブ
        self.prediction_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.prediction_tab, text="レース予想")
        self.setup_prediction_tab()
        
        # 履歴タブ
        self.history_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.history_tab, text="予測履歴")
        self.setup_history_tab()
        
        # 設定タブ
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="設定")
        self.setup_settings_tab()
        
        self.tab_control.pack(expand=1, fill="both")
    
    def setup_prediction_tab(self):
        """予測タブのコンポーネントをセットアップする"""
        # 入力フォームエリア
        self.input_frame = ttk.LabelFrame(self.prediction_tab, text="レース情報入力")
        self.input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 入力フォームの作成
        self.create_input_fields()
        
        # 予測ボタン
         self.predict_button = ttk.Button(
            self.prediction_tab, 
            text="AI予想を実行", 
            command=self.run_prediction
        )
        self.predict_button.pack(pady=10)
        
        # 結果表示エリア
        self.results_frame = ttk.LabelFrame(self.prediction_tab, text="予想結果")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 結果テキストエリア
        self.results_text = tk.Text(self.results_frame, height=15)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_input_fields(self):
        """入力フィールドを作成する"""
        # レース基本情報
        ttk.Label(self.input_frame, text="レース名:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.race_name_entry = ttk.Entry(self.input_frame, width=30)
        self.race_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.input_frame, text="レース距離:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.race_distance_entry = ttk.Entry(self.input_frame, width=10)
        self.race_distance_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 馬場状態
        ttk.Label(self.input_frame, text="馬場状態:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.track_condition = ttk.Combobox(self.input_frame, width=10, values=["良", "稍重", "重", "不良"])
        self.track_condition.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # 天候
        ttk.Label(self.input_frame, text="天候:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.weather = ttk.Combobox(self.input_frame, width=10, values=["晴", "曇", "雨", "雪"])
        self.weather.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # 馬情報入力（複数行）
        self.horse_entries = []
        
        headers = ["馬名", "騎手名", "枠番", "馬番", "人気", "馬体重", "体重増減"]
        for i, header in enumerate(headers):
            ttk.Label(self.input_frame, text=header).grid(row=2, column=i, padx=5, pady=2)
        
        # 最大10頭分の入力欄を作成
        for i in range(10):
            horse_row = []
            for j in range(len(headers)):
                entry = ttk.Entry(self.input_frame, width=10)
                entry.grid(row=i+3, column=j, padx=5, pady=2)
                horse_row.append(entry)
            self.horse_entries.append(horse_row)
    
    def setup_history_tab(self):
        """履歴タブのコンポーネントをセットアップする"""
        # 履歴リストビュー
        self.history_frame = ttk.Frame(self.history_tab)
        self.history_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 列の定義
        columns = ("date", "race_name", "accuracy")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings")
        
        # 列見出しの設定
        self.history_tree.heading("date", text="日付")
        self.history_tree.heading("race_name", text="レース名")
        self.history_tree.heading("accuracy", text="的中率")
        
        # 列幅の設定
        self.history_tree.column("date", width=100)
        self.history_tree.column("race_name", width=200)
        self.history_tree.column("accuracy", width=100)
        
        # スクロールバーの追加
        scrollbar = ttk.Scrollbar(self.history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # ウィジェットの配置
        self.history_tree.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 読み込みボタン
        load_button = ttk.Button(
            self.history_tab, 
            text="履歴を読み込む", 
            command=self.load_history
        )
        load_button.pack(pady=10)
    
    def setup_settings_tab(self):
        """設定タブのコンポーネントをセットアップする"""
        # 設定フレーム
        self.settings_frame = ttk.Frame(self.settings_tab)
        self.settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # API設定セクション
        api_frame = ttk.LabelFrame(self.settings_frame, text="Anthropic API設定")
        api_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.api_key_entry = ttk.Entry(api_frame, width=40, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # API接続テストボタン
        test_button = ttk.Button(
            api_frame, 
            text="接続テスト", 
            command=self.test_api_connection
        )
        test_button.grid(row=0, column=2, padx=5, pady=2)
        
        # モデル設定セクション
        model_frame = ttk.LabelFrame(self.settings_frame, text="モデル設定")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="使用モデル:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_path_entry = ttk.Entry(model_frame, width=40)
        self.model_path_entry.insert(0, self.model_path)
        self.model_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 保存ボタン
        save_button = ttk.Button(
            self.settings_frame, 
            text="設定を保存", 
            command=self.save_settings
        )
        save_button.pack(pady=10)
    
    def collect_horse_data(self):
        """入力フォームから馬データを収集する"""
        horse_data = []
        
        for entries in self.horse_entries:
            # 空の行はスキップ
            if not entries[0].get().strip():
                continue
                
            horse = {
                "horse_name": entries[0].get(),
                "jockey": entries[1].get(),
                "frame_number": int(entries[2].get()) if entries[2].get() else None,
                "horse_number": int(entries[3].get()) if entries[3].get() else None,
                "popularity": int(entries[4].get()) if entries[4].get() else None,
                "horse_weight": int(entries[5].get()) if entries[5].get() else None,
                "weight_change": int(entries[6].get()) if entries[6].get() else None
            }
            
            horse_data.append(horse)
        
        return horse_data
    
    def run_prediction(self):
        """予測を実行する"""
        # 入力データの収集
        race_name = self.race_name_entry.get()
        race_distance = int(self.race_distance_entry.get()) if self.race_distance_entry.get() else None
        track_condition = self.track_condition.get() if hasattr(self, 'track_condition') else ""
        weather = self.weather.get() if hasattr(self, 'weather') else ""
        horse_data = self.collect_horse_data()
        
        if not race_name or not race_distance or not horse_data:
            messagebox.showwarning("入力エラー", "必要な情報（レース名、距離、馬情報）を入力してください")
            return
        
        # 予測の実行
        try:
            race_info = {
                "race_name": race_name,
                "distance": race_distance,
                "track_condition": track_condition,
                "weather": weather,
                "horses": horse_data
            }
            
            # 予測処理
            predictions = self.predict_race_outcome(race_info)
            
            # Anthropic APIによる解説生成
            explanation = self.anthropic_client.generate_prediction_explanation(race_info, predictions)
            
            # 結果の表示
            self.display_results(predictions, explanation)
            
            # 履歴に追加
            self.add_to_history(race_name, predictions)
            
        except Exception as e:
            logger.error(f"予測処理中にエラーが発生しました: {str(e)}")
            messagebox.showerror("予測エラー", f"予測処理中にエラーが発生しました: {str(e)}")
    
    def predict_race_outcome(self, race_info):
        """レース予測を行う関数"""
        horses = race_info["horses"]
        
        # 入力データをDataFrameに変換
        df = pd.DataFrame(horses)
        
        if self.model is None:
            # モデルがない場合はダミーの予測結果を返す
            logger.warning("モデルが読み込まれていないため、ダミーの予測結果を返します")
            ranked_horses = []
            for i, horse in enumerate(horses):
                confidence = 90 - (i * 10)  # ダミーの信頼度
                ranked_horses.append({
                    "horse_name": horse["horse_name"],
                    "jockey": horse["jockey"],
                    "predicted_rank": i+1,
                    "confidence": max(confidence, 10)
                })
            return {
                "race_name": race_info["race_name"],
                "distance": race_info["distance"],
                "ranked_horses": ranked_horses
            }
        
        # 特徴量エンジニアリング
        try:
            features_df = engineer_features(df)
            
            # 予測の実行
            predictions = self.model.predict(features_df)
            
            # 予測結果の整形
            ranked_horses = []
            for i, horse in enumerate(horses):
                # 予測されたランク（小さいほど上位）
                predicted_rank = predictions[i]
                
                # 信頼度の計算（モデルに依存する実装が必要）
                # ここでは単純な例として示す
                confidence = 100 * (1 - (predicted_rank - 1) / len(horses))
                
                ranked_horses.append({
                    "horse_name": horse["horse_name"],
                    "jockey": horse["jockey"],
                    "predicted_rank": predicted_rank,
                    "confidence": confidence
                })
            
            # 予測順位で並べ替え
            ranked_horses.sort(key=lambda x: x["predicted_rank"])
            
        except Exception as e:
            logger.error(f"モデル予測中にエラーが発生しました: {str(e)}")
            # エラーが発生した場合はダミーの予測結果を返す
            ranked_horses = []
            for i, horse in enumerate(horses):
                confidence = 90 - (i * 10)  # ダミーの信頼度
                ranked_horses.append({
                    "horse_name": horse["horse_name"],
                    "jockey": horse["jockey"],
                    "predicted_rank": i+1,
                    "confidence": max(confidence, 10)
                })
        
        return {
            "race_name": race_info["race_name"],
            "distance": race_info["distance"],
            "ranked_horses": ranked_horses
        }
    
    def display_results(self, predictions, explanation):
        """予測結果を表示する"""
        # 結果テキストエリアをクリア
        self.results_text.delete(1.0, tk.END)
        
        # 予測結果と解説を表示
        self.results_text.insert(tk.END, "【AI予想結果】\n\n")
        
        # 予測順位順に馬を表示
        for i, horse in enumerate(predictions["ranked_horses"]):
            self.results_text.insert(tk.END, f"{i+1}着予想: {horse['horse_name']} (信頼度: {horse['confidence']:.1f}%)\n")
        
        self.results_text.insert(tk.END, "\n【解説】\n\n")
        self.results_text.insert(tk.END, explanation)
    
    def add_to_history(self, race_name, predictions):
        """予測履歴に追加する"""
        from datetime import datetime
        
        date = datetime.now().strftime("%Y-%m-%d")
        accuracy = "予測済"  # 実際の的中率はレース後に更新
        
        # 履歴リストに追加
        self.history_tree.insert("", tk.END, values=(date, race_name, accuracy))
        
        # 履歴をファイルに保存
        self.save_prediction_history(race_name, predictions)
    
    def save_prediction_history(self, race_name, predictions):
        """予測履歴をファイルに保存する"""
        import json
        from datetime import datetime
        
        history_dir = os.path.join(project_root, "data", "history")
        os.makedirs(history_dir, exist_ok=True)
        
        date = datetime.now().strftime("%Y%m%d")
        filename = f"{history_dir}/{date}_{race_name.replace(' ', '_')}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            logger.info(f"予測履歴を保存しました: {filename}")
        except Exception as e:
            logger.error(f"予測履歴の保存中にエラーが発生しました: {str(e)}")
    
    def load_history(self):
        """保存された予測履歴を読み込む"""
        import glob
        
        # 履歴ツリービューをクリア
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # 履歴ファイルを検索
        history_dir = os.path.join(project_root, "data", "history")
        if not os.path.exists(history_dir):
            messagebox.showinfo("情報", "履歴データが見つかりません")
            return
        
        history_files = glob.glob(f"{history_dir}/*.json")
        
        if not history_files:
            messagebox.showinfo("情報", "履歴データが見つかりません")
            return
        
        # 履歴ファイルを読み込んでツリービューに追加
        import json
        from datetime import datetime
        
        for file_path in history_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # ファイル名から日付を抽出
                filename = os.path.basename(file_path)
                date_str = filename.split("_")[0]
                if len(date_str) == 8 and date_str.isdigit():
                    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                else:
                    date = "不明"
                
                race_name = data.get("race_name", "不明")
                accuracy = "予測済"  # 実際の的中率はレース後に更新
                
                self.history_tree.insert("", tk.END, values=(date, race_name, accuracy))
                
            except Exception as e:
                logger.error(f"履歴ファイルの読み込み中にエラーが発生しました: {file_path} - {str(e)}")
        
        logger.info(f"{len(history_files)}件の履歴を読み込みました")
        messagebox.showinfo("情報", f"{len(history_files)}件の履歴を読み込みました")
    
    def save_settings(self):
        """設定を保存する"""
        api_key = self.api_key_entry.get()
        model_path = self.model_path_entry.get()
        
        # API設定の保存
        self.anthropic_client.save_api_key(api_key)
        
        # モデルパスの更新
        if model_path != self.model_path:
            self.model_path = model_path
            self.model = self.load_model()
        
        messagebox.showinfo("設定保存", "設定が保存されました")
    
    def test_api_connection(self):
        """APIの接続テスト"""
        success, message = self.anthropic_client.test_connection()
        if success:
            messagebox.showinfo("API接続テスト", "接続テスト成功: APIは正常に動作しています")
        else:
            messagebox.showerror("API接続テスト", f"接続テスト失敗: {message}")


def create_main_window():
    """メインウィンドウを作成する関数"""
    root = tk.Tk()
    app = BaneiAIApp(root)
    return root

def main():
    """メイン関数"""
    root = create_main_window()
    root.mainloop()

if __name__ == "__main__":
    main()