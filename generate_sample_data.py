from database import MongoDB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize MongoDB connection
        db = MongoDB()
        
        # Generate sample data
        success = db.generate_sample_data()
        
        if success:
            logger.info("Sample data generated successfully!")
        else:
            logger.error("Failed to generate sample data")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 