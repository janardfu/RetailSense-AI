from database import MongoDB

def test_mongodb():
    try:
        db = MongoDB()
        print("MongoDB class imported successfully")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb() 