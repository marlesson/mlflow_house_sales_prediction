# MLflow - House Sales Prediction

![house Logo](https://www.imovelweb.com.br/noticias/wp-content/uploads/2013/08/venda2.jpg)

# House Sales in King County, USA


Este conjunto de dados contém preços de imóveis para o Condado de King, que inclui Seattle. Inclui casas vendidas entre maio de 2014 e maio de 2015.

https://www.kaggle.com/harlfoxem/housesalesprediction

## Run

``
mlflow run https://github.com/marlesson/mlflow_house_sales_prediction.git
``


## Predict

``
curl -H "Content-Type: application/json" --data @body.json http://localhost:8080/ui/webapp/conf
``