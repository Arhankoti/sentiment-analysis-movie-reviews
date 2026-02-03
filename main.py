from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_data
from src.text_preprocessing import preprocess_text
from src.model import build_model
from src.evaluation import evaluate

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

model = build_model(preprocess_text)
model.fit(X_train, y_train)

evaluate(model, X_test, y_test)

print(model.predict(["The movie was boring and long"]))
