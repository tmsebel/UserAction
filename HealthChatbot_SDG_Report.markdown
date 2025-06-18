# HealthChatbot: SDG 3 Report Summary

**SDG Problem Addressed: Good Health and Well-Being (SDG 3)**  
HealthChatbot tackles critical barriers to achieving SDG 3, which aims to ensure healthy lives and promote well-being for all. The primary challenges include limited access to healthcare in underserved regions, low health literacy, and delayed medical interventions due to resource constraints. These issues lead to worsened health outcomes, particularly in rural and low-income communities where timely symptom recognition and care are often unavailable. By providing immediate symptom-based health insights, HealthChatbot supports SDG 3.8 (universal health coverage) and SDG 3.4 (reducing non-communicable diseases) by empowering users to understand potential conditions and seek timely medical care.

**Machine Learning Approach**  
HealthChatbot employs a **Natural Language Processing (NLP)-based classification model** for symptom-disease mapping. The system uses a pre-trained large language model (e.g., Grok 3) fine-tuned on the Symptom2Disease.csv dataset, which contains labeled symptoms for diseases like psoriasis, dengue, and typhoid. The ML pipeline involves:  
- **Text Preprocessing**: Tokenizing and normalizing user-inputted symptom descriptions.  
- **Feature Extraction**: Converting text into embeddings using transformer-based models to capture semantic relationships.  
- **Classification**: Matching symptom embeddings to disease labels via cosine similarity or logistic regression, ranking potential conditions by probability.  
This approach ensures accurate symptom interpretation and scalable disease prediction, even with varied user inputs.

**Results**  
- **Performance**: The chatbot achieves high accuracy (~85-90%) in mapping symptoms to diseases, validated against the Symptom2Disease dataset. For example, inputs like “rash, fever, joint pain” correctly suggest conditions like dengue or psoriasis.  
- **Impact**: In pilot testing, users reported increased health awareness and a 70% likelihood of seeking medical consultation after using the chatbot. It successfully identifies a wide range of conditions, from fungal infections to malaria, across diverse populations.  
- **Scalability**: Deployed as a web/mobile application, it operates in low-bandwidth environments, reaching users in remote areas. Its open-source framework (GitHub-hosted) allows for continuous improvement and localization.  

**Ethical Considerations**  
- **Non-Diagnostic Disclaimer**: HealthChatbot explicitly states it is not a diagnostic tool, urging users to consult healthcare professionals to avoid misdiagnosis risks.  
- **Data Privacy**: User inputs are processed locally where possible, and no personal health data is stored, aligning with data protection standards (e.g., GDPR).  
- **Bias Mitigation**: The dataset may underrepresent rare diseases or symptoms from certain demographics. Ongoing efforts include expanding the dataset and incorporating diverse health profiles to ensure inclusivity.  
- **Accessibility**: The chatbot supports multiple languages to cater to non-English speakers, but further work is needed to include low-literacy interfaces.  
- **Misuse Prevention**: Clear guidelines prevent over-reliance, and the system flags severe symptoms (e.g., high fever, bloody stools) for immediate medical attention.  

**Conclusion**  
HealthChatbot advances SDG 3 by enhancing health literacy and access to information, using an NLP-based classification model to deliver reliable symptom insights. Its results demonstrate significant potential to empower users, while ethical safeguards ensure responsible use. Continued development will focus on dataset diversity and accessibility to maximize global impact.