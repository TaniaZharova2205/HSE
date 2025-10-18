import io
import os
import logging
from datetime import datetime
import joblib


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
MY_NAME = "Tanya"
MY_SURNAME = "Zharova"
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}{MY_NAME[0]}_Final"

S3_BUCKET = Variable.get("S3_BUCKET")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")


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
    ts = datetime.utcnow().isoformat()
    logging.info(f"pipeline_start={ts}")
    context["ti"].xcom_push(key="pipeline_start", value=ts)


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


def train_and_log_mlflow(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    **context,
):
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    x_train = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/X_train.csv")
    y_train = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/y_train.csv")
    x_test = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/X_test.csv")
    y_test = s3_read_csv(hook, S3_BUCKET, f"{MY_SURNAME}/y_test.csv")
    
    logging.info(f"Данные загружены: X_train {x_train.shape}, X_test {x_test.shape}")
    
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    best_accuracy = 0
    best_model_name = ""
    best_run_id = ""
    with mlflow.start_run(run_name="tanoszha"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                model.fit(x_train, y_train.values.ravel())
                y_pred = model.predict(x_test)
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average='weighted')),
                    "recall": float(recall_score(y_test, y_pred, average='weighted')),
                    "f1": float(f1_score(y_test, y_pred, average='weighted'))
                }
            
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                signature = infer_signature(x_test, y_pred)
                mlflow.sklearn.log_model(model, name, signature=signature, input_example=x_train[:5])
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_model_name = name
                    best_run_id = mlflow.active_run().info.run_id
        REGISTERED_NAME = f"{best_model_name}_{MY_SURNAME}"
        try:
            client.get_registered_model(REGISTERED_NAME)
            print(f"Registered model {REGISTERED_NAME} already exists")
        except Exception:
            client.create_registered_model(REGISTERED_NAME)
            print(f"Created new registered model: {REGISTERED_NAME}")
        model_uri = f"runs:/{best_run_id}/{best_model_name}"

        mv = client.create_model_version(
            name=REGISTERED_NAME,
            source=model_uri,
            run_id=best_run_id
        )
        client.transition_model_version_stage(
            name=REGISTERED_NAME,
            version=mv.version,
            stage="Staging"
        )
        mlflow.log_params({
            "best_model": best_model_name,
            "best_accuracy": best_accuracy,
            "best_run_id": best_run_id,
            "registered_model": REGISTERED_NAME
        })
        mlflow.log_metrics({
            "best_accuracy": best_accuracy
        })
        context['ti'].xcom_push(key='best_run_id', value=best_run_id)
        context['ti'].xcom_push(key='best_model_name', value=best_model_name)
        context['ti'].xcom_push(key='best_accuracy', value=best_accuracy)

default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw3",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    init_pipeline = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": "https://storage.yandexcloud.net",
        },
    )
    serve_model = BashOperator(
        task_id="serve_model",
        bash_command=(
            "export PATH=$PATH:/home/airflow/.local/bin\n"
            "RUN_ID={{ ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_run_id') }}\n"
            "MODEL_NAME={{ ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_model_name') }}\n"
            "MODEL_URI=\"runs:/$RUN_ID/$MODEL_NAME\"\n"
            "echo \"Using MODEL_URI=$MODEL_URI\"\n"
            "mlflow models serve "
            "--model-uri $MODEL_URI "
            "--host 0.0.0.0 --port 5002 --no-conda &\n"
            "SERVER_PID=$!\n"
            "sleep 10\n"
            "curl -v --fail -X POST http://127.0.0.1:5002/invocations "
            "-H 'Content-Type: application/json' "
            "-d '{\"dataframe_split\": {\"columns\": [\"sepal length (cm)\", \"sepal width (cm)\", \"petal length (cm)\", \"petal width (cm)\"], \"data\": [[5.1, 3.5, 1.4, 0.2]]}}'\n"
            "STATUS=$?\n"
            "kill $SERVER_PID || true\n"
            "exit $STATUS\n"
        ),
        env={
            "MLFLOW_TRACKING_URI": Variable.get("MLFLOW_TRACKING_URI"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": "https://storage.yandexcloud.net",
        },
    )
    init_pipeline >> collect_data >> split_and_preprocess >> train_and_log_mlflow >> serve_model
