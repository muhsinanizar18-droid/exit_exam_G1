from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("best_shopping_model.pkl")

FEATURE_NAMES = [
    "Age", "Gender", "Occupation_Status", "Country_Region",
    "Monthly_Income_Range", "Shop_Most_Where", "Online_Shopping_Frequency",
    "Preferred_Platform", "Primary_Device", "Top_Spending_Category",
    "Review_Influence_Score", "Social_Ads_Influence_Score",
    "Price_Comparison_Frequency", "Discount_Importance_Score",
    "Follows_Brands_On_Social", "Likely_To_Buy_After_Ad_Score",
    "Makes_Shopping_List", "Impulse_Purchase_Frequency",
    "Return_Frequency", "Avg_Monthly_Spend_NonEssentials",
    "Preferred_Payment_Method", "Uses_BNPL_Installments",
    "Online_Shopping_Satisfaction_Score", "Trust_In_Online_Reviews_Score",
    "Regret_After_Purchase_Frequency", "Reason_Prefer_Online",
    "Reason_Prefer_InStore"
]

OPTIONS = {
    "Gender": ["Female", "Male", "Prefer not to say"],
    "Occupation_Status": ["Employed full-time", "Employed part-time", "Self-employed", "Student", "Unemployed"],
    "Country_Region": ["Brazil", "Colombia", "Egypt", "Germany", "India", "Indonesia", "Kenya", "Mexico", "Nigeria", "Pakistan", "Philippines", "Poland", "Slovakia", "South Africa", "Turkiye", "United Kingdom", "United States", "Vietnam"],
    "Monthly_Income_Range": ["Under $1,000", "$1,000-3,000", "$3,000-6,000", "$6,000+"],
    "Shop_Most_Where": ["Equal mix", "In-store only", "Online only"],
    "Online_Shopping_Frequency": ["Always", "Occasionally", "Often", "Rarely"],
    "Preferred_Platform": ["Brand websites", "Global marketplace (Amazon etc.)", "Local e-commerce", "Physical stores"],
    "Primary_Device": ["Laptop", "Smartphone", "Tablet"],
    "Top_Spending_Category": ["Beauty & personal care", "Books", "Clothing", "Electronics", "Groceries", "Home & kitchen", "Sports & fitness", "Travel", "Other"],
    "Follows_Brands_On_Social": ["No", "Yes"],
    "Makes_Shopping_List": ["Always", "Never", "Often", "Sometimes"],
    "Return_Frequency": ["Never", "Often", "Sometimes"],
    "Avg_Monthly_Spend_NonEssentials": ["Under $50", "$50-150", "$150-300", "$300-500", "$500+"],
    "Preferred_Payment_Method": ["Cash", "Credit Card", "Debit Card", "Digital wallet"],
    "Uses_BNPL_Installments": ["No, never", "Yes, occasionally", "Yes, regularly"],
    "Reason_Prefer_Online": ["Better prices", "Convenience", "More variety", "Not applicable"],
    "Reason_Prefer_InStore": ["No delivery wait", "Not applicable", "Trust", "Try product before buying"]
}

NUMERIC_FIELDS = {
    "Age", "Review_Influence_Score", "Social_Ads_Influence_Score",
    "Price_Comparison_Frequency", "Discount_Importance_Score",
    "Likely_To_Buy_After_Ad_Score", "Impulse_Purchase_Frequency",
    "Online_Shopping_Satisfaction_Score", "Trust_In_Online_Reviews_Score",
    "Regret_After_Purchase_Frequency"
}

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probabilities = None
    error = None

    if request.method == "POST":
        try:
            values = {}
            for field in FEATURE_NAMES:
                value = request.form.get(field)
                if value is None or value == "":
                    raise ValueError("Please complete every field.")
                values[field] = float(value) if field in NUMERIC_FIELDS else value

            input_data = pd.DataFrame([values], columns=FEATURE_NAMES)
            prediction = str(model.predict(input_data)[0])

            if hasattr(model, "predict_proba"):
                probabilities = sorted(
                    zip(model.classes_, model.predict_proba(input_data)[0]),
                    key=lambda item: item[1],
                    reverse=True
                )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        options=OPTIONS,
        prediction=prediction,
        probabilities=probabilities,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
