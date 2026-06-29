"""Recommendation text templates for lifestyle and preventive guidance."""

from __future__ import annotations


RECOMMENDATION_TEMPLATES: dict[str, dict[str, str]] = {
    "exercise": {
        "title": "Increase safe physical activity",
        "text": "Aim for regular moderate activity such as walking, cycling, or swimming as tolerated, and reduce long sitting periods.",
    },
    "diet": {
        "title": "Strengthen everyday nutrition",
        "text": "Emphasize vegetables, fruits, whole grains, lean proteins, and minimally processed foods while limiting sugary drinks and highly refined carbohydrates.",
    },
    "weight_management": {
        "title": "Support healthy weight management",
        "text": "Consider gradual lifestyle changes that support a healthy weight, such as balanced meals, portion awareness, and consistent activity.",
    },
    "blood_pressure": {
        "title": "Monitor blood pressure",
        "text": "Keep track of blood pressure and discuss elevated readings with a qualified healthcare professional.",
    },
    "smoking": {
        "title": "Reduce tobacco-related risk",
        "text": "If currently smoking, consider evidence-based cessation support such as counseling, quitlines, or clinician-guided resources.",
    },
    "alcohol": {
        "title": "Use alcohol cautiously",
        "text": "If drinking alcohol, keep intake within public health guidance and avoid binge drinking.",
    },
    "sleep": {
        "title": "Prioritize restorative sleep",
        "text": "Maintain a consistent sleep routine and seek professional guidance if poor sleep or daytime fatigue is persistent.",
    },
    "stress": {
        "title": "Manage stress and mental wellbeing",
        "text": "Use practical stress supports such as regular movement, relaxation techniques, social support, or professional counseling when needed.",
    },
    "preventive_screening": {
        "title": "Stay current with preventive screening",
        "text": "Discuss appropriate diabetes risk screening, cholesterol checks, and routine preventive care with a qualified healthcare professional.",
    },
    "follow_up": {
        "title": "Plan healthcare follow-up",
        "text": "Consider scheduling a non-urgent follow-up to review risk factors and prevention options with a qualified healthcare professional.",
    },
    "general_wellness": {
        "title": "Maintain protective routines",
        "text": "Continue health-supporting habits and review changes in risk factors over time.",
    },
}


POSITIVE_OBSERVATION_TEMPLATES: dict[str, str] = {
    "PhysActivity": "Physical activity is a positive lifestyle signal.",
    "Fruits": "Fruit consumption is a positive nutrition signal.",
    "Veggies": "Vegetable consumption is a positive nutrition signal.",
    "Smoker": "Not smoking is a positive health indicator.",
    "HvyAlcoholConsump": "Avoiding heavy alcohol consumption is a positive health indicator.",
    "HighBP": "No high blood pressure indicator was observed in the patient data.",
    "HighChol": "No high cholesterol indicator was observed in the patient data.",
}


FEATURE_CATEGORY_MAP: dict[str, list[str]] = {
    "BMI": ["weight_management", "diet", "exercise"],
    "HighBP": ["blood_pressure", "diet", "exercise"],
    "HighChol": ["diet", "preventive_screening"],
    "CholCheck": ["preventive_screening"],
    "PhysActivity": ["exercise"],
    "Fruits": ["diet"],
    "Veggies": ["diet"],
    "Smoker": ["smoking"],
    "HvyAlcoholConsump": ["alcohol"],
    "MentHlth": ["stress"],
    "PhysHlth": ["sleep", "follow_up"],
    "GenHlth": ["general_wellness", "follow_up"],
    "Age": ["preventive_screening", "follow_up"],
    "HeartDiseaseorAttack": ["follow_up", "preventive_screening"],
    "Stroke": ["follow_up", "preventive_screening"],
    "DiffWalk": ["exercise", "follow_up"],
    "NoDocbcCost": ["follow_up"],
    "AnyHealthcare": ["follow_up"],
    "Income": ["general_wellness"],
    "Education": ["general_wellness"],
}
