const tf = require('@tensorflow/tfjs');

exports.handler = async (event) => {
    bucketName = '100k-points-model'
    server = 'us-west-1'
    const MODEL_URL = `https://${bucketName}.s3.${server}.amazonaws.com/model.json`

    const model = await tf.loadLayersModel(MODEL_URL);

    let X = event.x;
    inputs = tf.tensor(X, [1, 51]);

    const result = model.predict(inputs);
    const y = (await result.array())[0][0];

    return {
        statusCode: 200,
        body: JSON.stringify(y)
    };
};