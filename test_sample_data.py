import logging
from database import MongoDB

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sample_data_generation():
    try:
        # Initialize MongoDB connection
        logger.info("Initializing MongoDB connection...")
        db = MongoDB()
        
        # Test database connection
        try:
            db.client.server_info()
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
        
        # Generate sample data
        logger.info("Starting sample data generation test...")
        success = db.generate_sample_data()
        
        if success:
            logger.info("Sample data generation completed successfully!")
            return True
        else:
            logger.error("Sample data generation failed!")
            return False
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    test_sample_data_generation() 