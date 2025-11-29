import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from app.config import settings


def init_firebase():
    """Initializes Firebase Admin SDK if not already initialized."""
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": settings.FIREBASE_PROJECT_ID,
                "private_key": settings.FIREBASE_PRIVATE_KEY.replace("\\n", "\n"),
                "client_email": settings.FIREBASE_CLIENT_EMAIL,
                "token_uri": "https://oauth2.googleapis.com/token"
            })

            firebase_admin.initialize_app(cred, {
                "storageBucket": f"{settings.FIREBASE_PROJECT_ID}.appspot.com",
            })

            print("🔥 Firebase initialized successfully")
        except Exception as e:
            print("❌ Firebase initialization failed:", e)
            raise e

    return firebase_admin


# Initialize on import
firebase_admin_app = init_firebase()

# Expose services for convenience
db = firestore.client()
auth_client = auth
storage_client = storage
