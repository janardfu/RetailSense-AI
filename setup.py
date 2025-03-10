from setuptools import setup, find_packages

setup(
    name="RetailSense",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.0',
        'numpy>=1.26.3',
        'scikit-learn>=1.3.2',
        'tensorflow>=2.15.0',
        'statsmodels>=0.14.1',
        'streamlit>=1.31.1',
        'plotly>=5.18.0',
        'fastapi>=0.110.0',
        'uvicorn>=0.27.1',
        'python-dotenv>=1.0.1',
        'pymongo>=4.6.1',
    ],
) 