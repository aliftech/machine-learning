import sys

@app.post('/')
def main(data: request_body):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    index = clf.predict(test_data)[0]
    return {'class': iris.target_names[index]}

sys.modules[__name__] = main