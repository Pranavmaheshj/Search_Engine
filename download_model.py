from sentence_transformers import SentenceTransformer

print("Starting model download...")
print("This may take several minutes depending on your internet speed. Please be patient.")

# This line will download and cache the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("\n✅ Model download complete!")
print("The model is now cached on your computer.")
print("You can now start the main application.")