import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import base64

# Import Vertex AI SDK
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

CLINICS_LIST = """
"الباطنة-والجهاز-الهضمي-والكبد", "مسالك", "باطنة-عامة", "غدد-صماء-وسكر", "القلب-والإيكو",
"السونار-والدوبلكس", "جراحة-التجميل", "عظام", "جلدية-وليزر"
"""

# قم بتهيئة Vertex AI مرة واحدة عند بدء تشغيل التطبيق
# PROJECT_ID و LOCATION سيتم الكشف عنهما تلقائيًا في Cloud Run
try:
    vertexai.init(project=os.environ.get("GCP_PROJECT"), location=os.environ.get("GCP_REGION", "europe-west1"))
    # تأكد من أن النموذج متاح في هذه المنطقة
    # يمكنك تبديل النموذج إذا لم يكن gemini-1.5-flash متاحًا
    # مثال: model_name = "gemini-1.5-flash-001" أو "gemini-1.0-pro"
    global_model = GenerativeModel("gemini-1.5-flash") # استخدم اسم النموذج مباشرة هنا
    print(f"Vertex AI initialized and model {global_model._model_id} loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Vertex AI or load model: {e}")
    global_model = None # تأكد من أن النموذج ليس متاحًا إذا فشلت التهيئة

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    if global_model is None:
        return jsonify({"error": "Server is not configured correctly. AI model failed to load."}), 500

    try:
        data = request.get_json()
        symptoms = data.get('symptoms')
        if not symptoms:
            return jsonify({"error": "Missing symptoms"}), 400
        
        # لا حاجة لـ API_KEY هنا إذا كنت تستخدم Vertex AI SDK داخل Cloud Run
        # حيث يتم التعامل مع المصادقة عبر Service Account الافتراضي
        
        # لا داعي لإعادة تهيئة النموذج في كل طلب
        # model = GenerativeModel('gemini-1.5-flash')

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7, # يمكنك ضبط درجة الحرارة لتحكم في عشوائية الاستجابة
        )

        prompt = f"""
        أنت مساعد طبي خبير ومحترف في مستشفى كبير. مهمتك هي تحليل شكوى المريض بدقة واقتراح أفضل عيادتين بحد أقصى من قائمة العيادات المتاحة.
        قائمة معرفات (IDs) العيادات المتاحة هي: [{CLINICS_LIST}]
        شكوى المريض: "{symptoms}"
        
        المطلوب منك:
        1.  حدد العيادة الأساسية الأكثر احتمالاً بناءً على الأعراض الرئيسية في الشكوى.
        2.  اشرح للمريض بلغة عربية بسيطة ومباشرة **لماذا** قمت بترشيح هذه العيادة.
        3.  إذا كان هناك احتمال آخر قوي، حدد عيادة ثانوية واشرح أيضاً لماذا قد تكون خياراً جيداً.
        4.  إذا كانت الشكوى غامضة جداً (مثل "أنا متعب")، قم بترشيح "باطنة-عامة" واشرح أن الفحص العام هو أفضل نقطة بداية.
        5.  ردك **يجب** أن يكون بصيغة JSON فقط، بدون أي نصوص أو علامات قبله أو بعده. يجب أن يكون على هذا الشكل بالضبط:
        {{
          "recommendations": [
            {{ "id": "ID_العيادة", "reason": "شرح سبب الاختيار." }}
          ]
        }}
        """
        
        response = global_model.generate_content(prompt, generation_config=generation_config)
        
        # تأكد من أن الاستجابة تحتوي على نص قبل محاولة تحليلها
        if response.text:
            json_response = json.loads(response.text)
            return jsonify(json_response)
        else:
            print("ERROR: Gemini API returned empty response.")
            return jsonify({"error": "Gemini API returned an empty response."}), 500
        
    except Exception as e:
        print(f"ERROR in /api/recommend: {str(e)}")
        # ارجع رسالة خطأ أكثر تفصيلاً للمساعدة في Debugging
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

