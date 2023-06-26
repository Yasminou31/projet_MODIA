import argparse
import joblib
import gradio as gr

# hide warnings
import warnings
warnings.filterwarnings('ignore')



def analyze_review(review_text):
    # Create prediction-label mapper
    labels = {0: 'Negative', 1: 'Positive'}
    # make prediction
    prediction = model.predict([review_text])
    # return the prediction
    print(prediction)
    return labels[int(prediction[0])]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='sentiment_model.pkl')
    args = parser.parse_args()

    # get the model weights path and the path of the test_script.csv
    model_path = args.model_path
    model = joblib.load(model_path)

    # Interface
    gr.Interface(fn=analyze_review, 
                inputs="textbox", 
                outputs=gr.outputs.Label(num_top_classes=1),
                live=True,
                description="Write a review and the model will predict if it is positive or negative.",
                ).launch(debug=True, share=True, server_name="0.0.0.0", server_port=8080)

    