import re
import json
import os
import datetime
import csv
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Symptom synonyms for flexible matching
SYMPTOM_SYNONYMS = {
    "fever": ["fever", "high temperature", "hot", "burning up"],
    "headache": ["headache", "head hurts", "pain in head", "migraine"],
    "cough": ["cough", "coughing", "hack"],
    "stomach pain": ["stomach pain", "stomach hurts", "belly pain", "abdominal pain"],
    "diarrhea": ["diarrhea", "loose stools", "runny tummy"],
    "sore throat": ["sore throat", "throat pain", "hurts to swallow"],
    "fatigue": ["fatigue", "tired", "exhausted", "worn out"]
}

# Mock data as fallback
MOCK_DATA = {
    "fever": {
        "follow_up": ["Is the fever above 100.4°F (38°C)? (yes/no)", "Do you have a cough? (yes/no)"],
        "diagnoses": {
            ("high", "yes"): {
                "condition": "Possible Flu",
                "first_aid": "Rest, stay hydrated, take acetaminophen.",
                "emergency": "Seek care if fever persists >3 days or breathing issues occur."
            },
            ("high", "no"): {
                "condition": "Possible Malaria",
                "first_aid": "Stay hydrated, avoid exertion.",
                "emergency": "Seek immediate care for malaria testing in tropical areas."
            },
            ("low", "no"): {
                "condition": "Mild Fever",
                "first_aid": "Rest, drink fluids.",
                "emergency": "Monitor for worsening symptoms."
            }
        }
    },
    "headache": {
        "follow_up": ["Is the headache severe or sudden? (yes/no)", "Any vision changes? (yes/no)"],
        "diagnoses": {
            ("yes", "yes"): {
                "condition": "Possible Migraine or Serious Issue",
                "first_aid": "Rest in a dark room, avoid screens.",
                "emergency": "Seek immediate care for sudden severe headache."
            },
            ("no", "no"): {
                "condition": "Tension Headache",
                "first_aid": "Rest, hydrate, consider ibuprofen.",
                "emergency": "Seek care if headache persists >2 days."
            }
        }
    },
    "cough": {
        "follow_up": ["Is the cough dry or productive (phlegm)? (dry/productive)", "Any shortness of breath? (yes/no)"],
        "diagnoses": {
            ("dry", "no"): {
                "condition": "Possible Allergies or Cold",
                "first_aid": "Use a humidifier, drink warm fluids.",
                "emergency": "Seek care if cough persists >2 weeks."
            },
            ("productive", "yes"): {
                "condition": "Possible Pneumonia",
                "first_aid": "Rest, stay hydrated.",
                "emergency": "Seek immediate care for breathing difficulty."
            }
        }
    },
    "stomach pain": {
        "follow_up": ["Do you have diarrhea? (yes/no)", "Any nausea or vomiting? (yes/no)"],
        "diagnoses": {
            ("yes", "yes"): {
                "condition": "Possible Gastroenteritis",
                "first_aid": "Stay hydrated with oral rehydration solution.",
                "emergency": "Seek care if dehydration signs (e.g., dry mouth) appear."
            },
            ("no", "no"): {
                "condition": "Indigestion",
                "first_aid": "Avoid heavy meals, try antacids.",
                "emergency": "Seek care if pain is severe or persistent."
            }
        }
    },
    "diarrhea": {
        "follow_up": ["Is it watery or bloody? (watery/bloody)", "Any fever? (yes/no)"],
        "diagnoses": {
            ("bloody", "yes"): {
                "condition": "Possible Dysentery",
                "first_aid": "Stay hydrated, avoid solid food.",
                "emergency": "Seek immediate care for bloody diarrhea."
            },
            ("watery", "no"): {
                "condition": "Mild Diarrhea",
                "first_aid": "Drink oral rehydration solution.",
                "emergency": "Seek care if diarrhea lasts >2 days."
            }
        }
    },
    "sore throat": {
        "follow_up": ["Do you have a fever? (yes/no)", "Is it painful to swallow? (yes/no)"],
        "diagnoses": {
            ("yes", "yes"): {
                "condition": "Possible Strep Throat",
                "first_aid": "Gargle with warm saltwater, stay hydrated.",
                "emergency": "Seek care for a throat swab to test for bacterial infection."
            },
            ("no", "no"): {
                "condition": "Viral Pharyngitis",
                "first_aid": "Rest, drink warm fluids, use throat lozenges.",
                "emergency": "Seek care if symptoms persist >5 days."
            }
        }
    },
    "fatigue": {
        "follow_up": ["Are you getting less than 6 hours of sleep nightly? (yes/no)", "Any recent weight loss? (yes/no)"],
        "diagnoses": {
            ("yes", "no"): {
                "condition": "General Tiredness",
                "first_aid": "Improve sleep hygiene, maintain a regular schedule.",
                "emergency": "Seek care if fatigue persists despite rest."
            },
            ("no", "yes"): {
                "condition": "Possible Anemia or Other Causes",
                "first_aid": "Eat a balanced diet, rest.",
                "emergency": "Seek medical evaluation for unexplained weight loss."
            }
        }
    }
}

def fetch_kaggle_dataset():
    """Fetch and process Kaggle dataset to generate health_data.csv."""
    # Download latest version from KaggleHub as an additional data source
    # path = kagglehub.dataset_download("tanishchavaan/disease-prediction-medical-dataset")
    dataset = "tanishchavaan/disease-prediction-medical-dataset"
    #dataset = "niyarrbarman/symptom2disease"
    output_dir = "kaggle_data"
    output_csv = "health_data.csv"
    data_source = "Kaggle API"

    try:
        # Check kaggle.json
        kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        if not os.path.exists(kaggle_json):
            raise FileNotFoundError(f"kaggle.json not found at {kaggle_json}. Please set up Kaggle API credentials.")
        logger.info(f"Found kaggle.json at {kaggle_json}")

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authenticated successfully")

        # Verify dataset exists
        try:
            api.dataset_metadata(dataset, path=output_dir)
            logger.info(f"Dataset {dataset} exists and is accessible")
        except Exception as meta_error:
            raise Exception(f"Dataset {dataset} not found or inaccessible: {str(meta_error)}")

        # Download dataset with retry
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Downloading dataset {dataset} to {output_dir}")
        try:
            api.dataset_download_files(dataset, path=output_dir, unzip=True, quiet=False)
        except Exception as download_error:
            if "403" in str(download_error):
                raise Exception(
                    f"403 Forbidden: Please visit https://www.kaggle.com/datasets/{dataset}, "
                    "log in, and accept the dataset rules to enable API access."
                )
            raise download_error

        # Find CSV file in downloaded dataset
        csv_file = None
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_file = os.path.join(root, file)
                    logger.info(f"Found CSV file: {csv_file}")
                    break
            if csv_file:
                break

        if not csv_file:
            raise FileNotFoundError("No CSV file found in Kaggle dataset")

        # Process dataset to match health_data.csv format
        with open(csv_file, "r", encoding="utf-8") as f_in, open(output_csv, "w", encoding="utf-8", newline="") as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.writer(f_out)
            writer.writerow(["symptom", "condition", "follow_up_1", "follow_up_2", "answer_1", "answer_2", "first_aid", "emergency"])

            for row in reader:
                symptom_desc = row["Symptom"].lower()
                condition = row["Disease"]

                # Map symptom description to chatbot symptoms
                mapped_symptom = None
                for symptom, synonyms in SYMPTOM_SYNONYMS.items():
                    for synonym in synonyms:
                        if synonym in symptom_desc:
                            mapped_symptom = symptom
                            break
                    if mapped_symptom:
                        break

                if not mapped_symptom:
                    logger.warning(f"Skipping unmapped symptom: {symptom_desc}")
                    continue

                # Use MOCK_DATA for follow-up questions and advice
                if mapped_symptom in MOCK_DATA:
                    for answers, diag in MOCK_DATA[mapped_symptom]["diagnoses"].items():
                        writer.writerow([
                            mapped_symptom,
                            condition,
                            MOCK_DATA[mapped_symptom]["follow_up"][0],
                            MOCK_DATA[mapped_symptom]["follow_up"][1],
                            answers[0],
                            answers[1],
                            diag["first_aid"],
                            diag["emergency"]
                        ])
                    logger.info(f"Processed symptom {mapped_symptom} with condition {condition}")

        logger.info(f"Generated {output_csv} from Kaggle dataset")
        return data_source

    except Exception as e:
        logger.error(f"Error fetching Kaggle dataset: {str(e)}")
        logger.info(
            f"Manual fallback: Download the dataset from https://www.kaggle.com/datasets/{dataset}, "
            "extract the CSV, and place it in kaggle_data/ as Symptom2Disease.csv"
        )
        data_source = "CSV" if os.path.exists(output_csv) else "Mock"
        return data_source

def load_health_data():
    """Load health data from CSV or return mock data as fallback."""
    data_file = "health_data.csv"
    health_data = {}
    data_source = "CSV"

    if os.path.exists(data_file):
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symptom = row["symptom"].lower()
                    if symptom not in health_data:
                        health_data[symptom] = {"follow_up": [], "diagnoses": {}}
                    follow_up = [row["follow_up_1"], row["follow_up_2"]]
                    if not health_data[symptom]["follow_up"]:
                        health_data[symptom]["follow_up"] = follow_up
                    answers_key = (row["answer_1"], row["answer_2"])
                    health_data[symptom]["diagnoses"][answers_key] = {
                        "condition": row["condition"],
                        "first_aid": row["first_aid"],
                        "emergency": row["emergency"]
                    }
            logger.info(f"Loaded health data from {data_file}")
            return health_data, data_source
        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}. Using mock data.")
            data_source = "Mock"
    else:
        logger.warning(f"{data_file} not found. Attempting to fetch from Kaggle.")
        data_source = fetch_kaggle_dataset()
        if data_source == "CSV" and os.path.exists(data_file):
            return load_health_data()  # Retry loading CSV
        data_source = "Mock"

    logger.info("Using mock data as final fallback")
    return MOCK_DATA, data_source

def adjust_diagnosis(symptom, diagnosis, geo_data, lifestyle, age):
    """Adjust diagnosis based on geo-data, lifestyle, and age."""
    if geo_data.get("region") == "tropical" and symptom in ["fever", "diarrhea", "sore throat"]:
        diagnosis["condition"] += " or Possible Tropical Infection"
        diagnosis["emergency"] += " Consider testing for regional diseases (e.g., malaria, dengue)."
    if lifestyle.get("smoker") and symptom in ["cough", "fever", "sore throat"]:
        diagnosis["emergency"] += " Smokers may have higher risk of respiratory issues."
    if age < 5 and symptom in ["fever", "diarrhea"]:
        diagnosis["first_aid"] += " Consult a pediatrician for child-safe treatments."
        diagnosis["emergency"] += " Children under 5 are at higher risk; seek care promptly."
    if age > 65 and symptom in ["fever", "cough", "fatigue"]:
        diagnosis["emergency"] += " Elderly patients may have higher risks; seek care promptly."
    return diagnosis

def get_user_input(prompt, valid_answers=None, is_numeric=False):
    """Get sanitized user input with optional validation."""
    while True:
        response = input(prompt).lower().strip()
        if not response:
            print("Please provide a valid input.")
            continue
        if is_numeric:
            try:
                value = int(response)
                if valid_answers and (value < valid_answers[0] or value > valid_answers[1]):
                    print(f"Please enter a number between {valid_answers[0]} and {valid_answers[1]}.")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number.")
                continue
        if valid_answers and response not in valid_answers:
            print(f"Please answer one of: {', '.join(valid_answers)}.")
            continue
        return response

def recognize_symptom(user_input):
    """Simulate BERT intent recognition with synonym and partial matching."""
    for symptom, synonyms in SYMPTOM_SYNONYMS.items():
        for synonym in synonyms:
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, user_input, re.IGNORECASE):
                return symptom
    return None

def log_interaction(user_name, user_profile, user_input, symptom, follow_up_answers, diagnosis, data_source):
    """Log interaction to a user-specific file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"chat_history_{user_name}.txt")
    
    log_entry = f"\n=== Session at {timestamp} ===\n"
    log_entry += f"Data Source: {data_source}\n"
    log_entry += f"User Profile: Age={user_profile['age']}, Smoker={user_profile['lifestyle']['smoker']}, Region={user_profile['geo_data']['region']}\n"
    log_entry += f"User Input: {user_input}\n"
    log_entry += f"Recognized Symptom: {symptom}\n"
    log_entry += "Follow-up Answers:\n"
    for question, answer in zip(MOCK_DATA[symptom]["follow_up"], follow_up_answers):
        log_entry += f"  Q: {question} A: {answer}\n"
    log_entry += f"Diagnosis:\n"
    log_entry += f"  Condition: {diagnosis['condition']}\n"
    log_entry += f"  First Aid: {diagnosis['first_aid']}\n"
    log_entry += f"  Emergency: {diagnosis['emergency']}\n"
    log_entry += "=" * 50 + "\n"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def main():
    # Load health data
    health_data, data_source = load_health_data()
    
    # Get user's name
    user_name = get_user_input("Please enter your name: ")
    
    # Get user profile
    age = get_user_input("Please enter your age (1-120): ", is_numeric=True, valid_answers=(1, 120))
    smoker = get_user_input("Do you smoke? (yes/no): ", valid_answers=["yes", "no"])
    region = get_user_input("Are you in a tropical region? (tropical/non-tropical): ", valid_answers=["tropical", "non-tropical"])
    
    user_profile = {
        "age": age,
        "gender": "unknown",
        "lifestyle": {"smoker": smoker == "yes"},
        "geo_data": {"region": region}
    }
    
    # Personalized welcome message
    print(f"\nWelcome, {user_name}, to the Health Diagnostic Chatbot!")
    print(f"Profile: Age={age}, Smoker={smoker}, Region={region}")
    print(f"Data Source: {data_source}")
    print("This tool helps diagnose common symptoms. Supported symptoms are: fever, headache, cough, stomach pain, diarrhea, sore throat, fatigue.")
    print("Enter your symptoms (e.g., 'I have a fever' or 'my head hurts') or type 'exit' to quit.\n")
    print("Your interactions will be logged to a file for record-keeping.\n")

    while True:
        user_input = get_user_input("Your symptoms: ")
        if user_input == "exit":
            print(f"Goodbye, {user_name}! Your chat history is saved in logs/chat_history_{user_name}.txt.")
            break

        symptom = recognize_symptom(user_input)
        if not symptom:
            print("Sorry, I didn't recognize that symptom. Try rephrasing (e.g., 'my head hurts' for headache) or use one of: fever, headache, cough, stomach pain, diarrhea, sore throat, fatigue.")
            continue

        print(f"Understood, you mentioned {symptom}. Please answer the following questions.")
        follow_up_answers = []
        for question in health_data[symptom]["follow_up"]:
            if "(yes/no)" in question:
                valid_answers = ["yes", "no"]
                answer = get_user_input(question + " ", valid_answers=valid_answers)
                answer = "high" if answer == "yes" and symptom == "fever" else answer
                follow_up_answers.append(answer)
            elif "(dry/productive)" in question:
                valid_answers = ["dry", "productive"]
                answer = get_user_input(question + " ", valid_answers=valid_answers)
                follow_up_answers.append(answer)
            elif "(watery/bloody)" in question:
                valid_answers = ["watery", "bloody"]
                answer = get_user_input(question + " ", valid_answers=valid_answers)
                follow_up_answers.append(answer)

        # Get diagnosis
        answers_key = tuple(follow_up_answers)
        diagnosis = health_data[symptom]["diagnoses"].get(answers_key, {
            "condition": "Unknown",
            "first_aid": "Rest and monitor symptoms.",
            "emergency": "Seek care if symptoms worsen."
        })
        diagnosis = adjust_diagnosis(symptom, diagnosis, user_profile["geo_data"], user_profile["lifestyle"], user_profile["age"])

        # Log interaction
        log_interaction(user_name, user_profile, user_input, symptom, follow_up_answers, diagnosis, data_source)

        # Display results
        print("\nDiagnosis Result:")
        print(f"Condition: {diagnosis['condition']}")
        print(f"First Aid: {diagnosis['first_aid']}")
        print(f"Emergency: {diagnosis['emergency']}\n")

if __name__ == "__main__":
    main()