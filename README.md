# 🐴 ばんえい競馬AI予想システム

![52cfe946f19a520aadb8095713a69dbe](https://github.com/user-attachments/assets/8d07dbdc-32ce-46e5-a711-9875cb90d9a3)

## こんにちは！👋


初めて作成した競馬予想AIアプリケーションをシェアします！これは北海道帯広市で行われる「ばんえい競馬」（重い荷物を引く馬のレース）の結果を予測するAIシステムです。
まだ開発途中(WIP)ですが、ぜひ試してみてください！

## ✨ できること

- 🔍 レースURLを入力するだけで、自動的に出走馬の情報を取得
- 🤖 AIが馬の着順を予測し、分かりやすい解説付きで表示
- ⚙️ 設定タブからAPIキーや使用モデルのカスタマイズが可能

## 🚀 使い方

1. アプリを起動する
2. 「レース予想」タブでレースURLを入力し「情報取得」ボタンをクリック
   - または手動で馬の情報を入力することもできます！
3. 「AI予想を実行」ボタンをクリックすると予測結果が表示されます

## ⚠️ 注意事項

- これは私の初めての本格的なアプリで、現在も開発中です
- 予測はあくまで参考程度に！実際の馬券購入は自己責任でお願いします
- 時々エラーが出ることがありますが、温かい目で見守ってください😅

## 🛠️ インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/あなたのユーザー名/banei-ai-predictor.git
cd banei-ai-predictor

# 必要なライブラリをインストール
pip install -r requirements.txt

# アプリを実行
python main.py
