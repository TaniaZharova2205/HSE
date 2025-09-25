from datetime import datetime
import io
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import json

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = "Tanya"
MY_SURNAME = "Zharova"

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


# -----------------
# Таски
# -----------------


def init_pipeline(**context):
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)


def collect_data(**context):
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    logging.info(f"Прочитан файл: iris.csv")
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    key = f"{MY_SURNAME}/iris.csv"
    s3_write_csv(hook, df, S3_BUCKET, key)
    context["ti"].xcom_push(key="collect_data", value=df.shape)

def split_and_preprocess(**context):
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    key = f"{MY_SURNAME}/iris.csv"
    df = s3_read_csv(hook, S3_BUCKET, key)
    logging.info(f"Считали файл: iris.csv")
    label_encoder = LabelEncoder()
    df['species_encoded'] = label_encoder.fit_transform(df['species'])
    classes = label_encoder.classes_
    encoding_mapping = {class_name: code for code, class_name in enumerate(classes)}
    logging.info(f"LabelEncoder mapping: {encoding_mapping}")

    x = df.drop(['species', 'species_encoded'], axis=1)
    y = df['species_encoded']

    logging.info(f"Уникальные классы в y: {y.unique()}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    logging.info("Данные успешно разделены:")
    logging.info(f"X_train: {x_train.shape}")
    logging.info(f"X_test: {x_test.shape}")
    logging.info(f"y_train: {y_train.shape}")
    logging.info(f"y_test: {y_test.shape}")

    xtrain_key = f"{MY_SURNAME}/X_train.csv"
    s3_write_csv(hook, x_train, S3_BUCKET, xtrain_key)
    ytrain_key = f"{MY_SURNAME}/y_train.csv"
    s3_write_csv(hook, y_train, S3_BUCKET, ytrain_key)
    logging.info(f"Train данные сохранены")

    xtest_key = f"{MY_SURNAME}/X_test.csv"
    s3_write_csv(hook, x_test, S3_BUCKET, xtest_key)
    ytest_key = f"{MY_SURNAME}/y_test.csv"
    s3_write_csv(hook, y_test, S3_BUCKET, ytest_key)
    logging.info(f"Test данные сохранены")

    encoder_data = {
        'classes': label_encoder.classes_.tolist(),
        'mapping': {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    }

    mapping_df = pd.DataFrame(list(encoder_data['mapping'].items()),
                              columns=['species', 'encoded_value'])
    mapping_key = f"{MY_SURNAME}/label_encoder_mapping.csv"
    s3_write_csv(hook, mapping_df, S3_BUCKET, mapping_key)
    logging.info(f"Маппинг LabelEncoder сохранен")
    encoder_buf = io.BytesIO()
    joblib.dump(label_encoder, encoder_buf)
    encoder_buf.seek(0)
    encoder_key = f"{MY_SURNAME}/label_encoder.joblib"
    hook.get_conn().upload_fileobj(encoder_buf, S3_BUCKET, encoder_key)
    logging.info(f"LabelEncoder сохранен")
    context['task_instance'].xcom_push(key='X_train_shape', value=str(x_train.shape))
    context['task_instance'].xcom_push(key='X_test_shape', value=str(x_test.shape))
    context['task_instance'].xcom_push(key='classes', value=encoder_data['classes'])

    logging.info("Препроцессинг с LabelEncoder успешно завершен!")



def train_model(**context):
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    x_train = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/X_train.csv")
    y_train = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/y_train.csv")
    model = LogisticRegression()
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Начало обучение модели: {start_ts}")
    model.fit(x_train, y_train)
    end_ts = datetime.utcnow().isoformat()
    logging.info(f"Конец обучения модели: {end_ts}")
    context["ti"].xcom_push(key="start_time", value=start_ts)
    context["ti"].xcom_push(key="end_time", value=end_ts)
    encoder_buf = io.BytesIO()
    joblib.dump(model, encoder_buf)
    encoder_buf.seek(0)
    hook.get_conn().upload_fileobj(encoder_buf, S3_BUCKET, f"{MY_SURNAME}/model.joblib")
    logging.info(f"Обучили модель!!!")


def collect_metrics_model(**context):
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    model_buf = io.BytesIO()
    hook.get_conn().download_fileobj(S3_BUCKET, f"{MY_SURNAME}/model.joblib", model_buf)
    model_buf.seek(0)
    model = joblib.load(model_buf)
    x_test = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/X_test.csv")
    y_test = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/y_test.csv")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logging.info("Метрики модели:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    context["ti"].xcom_push(key="accuracy_score", value=accuracy)
    context["ti"].xcom_push(key="precision_score", value=precision)
    context["ti"].xcom_push(key="recall_score", value=recall)
    context["ti"].xcom_push(key="f1_score", value=f1)

    metrics_data = {
        "accuracy_score": float(accuracy),
        "precision_score": float(precision),
        "recall_score": float(recall),
        "f1_score": float(f1),
        "model_name": "LogisticRegression",
        "dataset": "iris",
    }
    accuracy_buf = io.BytesIO()

    json_str = json.dumps(metrics_data, ensure_ascii=False, indent=4)
    accuracy_buf.write(json_str.encode('utf-8'))
    accuracy_buf.seek(0)

    hook.get_conn().upload_fileobj(accuracy_buf, S3_BUCKET, S3_KEY_MODEL_METRICS)

    logging.info("Сбор метрик модели завершен успешно!")

def collect_metrics_pipeline(**context):
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    pipeline_start = context["ti"].xcom_pull(task_ids="init_pipeline", key="pipeline_start")
    train_start = context["ti"].xcom_pull(task_ids="train_model", key="start_time")
    train_end = context["ti"].xcom_pull(task_ids="train_model", key="end_time")
    pipeline_end = datetime.utcnow().isoformat()
    start_dt = datetime.fromisoformat(pipeline_start)
    end_dt = datetime.fromisoformat(pipeline_end)
    train_start_dt = datetime.fromisoformat(train_start)
    train_end_dt = datetime.fromisoformat(train_end)

    pipeline_duration_seconds = (end_dt - start_dt).total_seconds()
    training_duration_seconds = (train_end_dt - train_start_dt).total_seconds()

    pipeline_duration_str = f"{pipeline_duration_seconds:.2f}"
    training_duration_str = f"{training_duration_seconds:.2f}"

    pipeline_metrics = {
        "pipeline_info": {
            "pipeline_name": "hw_1",
            "owner": f"{MY_NAME} {MY_SURNAME}",
            "dag_id": "hw_1"
        },
        "timestamps": {
            "pipeline_start": pipeline_start,
            "pipeline_end": pipeline_end,
            "training_start": train_start,
            "training_end": train_end
        },
        "durations": {
            "total_pipeline_duration_seconds": pipeline_duration_seconds,
            "total_pipeline_duration_readable": pipeline_duration_str,
            "training_duration_seconds": training_duration_seconds,
            "training_duration_readable": training_duration_str
        }
    }

    logging.info(f"Общее время выполнения: {pipeline_duration_str}")
    logging.info(f"Время обучения модели: {training_duration_str}")

    pipeline_buf = io.BytesIO()
    json_str = json.dumps(pipeline_metrics, ensure_ascii=False, indent=4)
    pipeline_buf.write(json_str.encode('utf-8'))
    pipeline_buf.seek(0)

    hook.get_conn().upload_fileobj(pipeline_buf, S3_BUCKET, S3_KEY_PIPELINE_METRICS)

    context["ti"].xcom_push(key="pipeline_duration", value=pipeline_duration_str)
    context["ti"].xcom_push(key="training_duration", value=training_duration_str)
    logging.info("Спасибо за внимание!")


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="zharova_hw1",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)
    t3 = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)
    t5 = PythonOperator(task_id="collect_metrics_model", python_callable=collect_metrics_model)
    t6 = PythonOperator(task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
