const tf = require('@tensorflow/tfjs');

exports.handler = async (event) => {
    bucketName = '100k-points-model'
    const MODEL_URL = `https://${bucketName}.s3.amazonaws.com/model.json`

    const model = await tf.loadLayersModel(MODEL_URL);

    let X = parseFloat(event.X);
    inputs = tf.tensor(X);

    const result = model.predict(inputs);
    const y = (await result.array())[0][0];

    return {
        statusCode: 200,
        body: JSON.stringify(y)
    };
};