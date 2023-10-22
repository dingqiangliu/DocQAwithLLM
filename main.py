import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        nargs='?',
                        type=str,
                        default='',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    dbqa = setup_dbqa()
    
    # query loop
    query = args.input
    if len(query) == 0:
        query = input('Enter the query: ').strip() 
    while query != '\\q':
        start = timeit.default_timer()
        response = dbqa({'query': query})
        end = timeit.default_timer()
    
        print(f'\nAnswer: {response["result"]}')
        print('='*50)
    
        # Process source documents
        source_docs = response['source_documents']
        for i, doc in enumerate(source_docs):
            print(f'\nSource Document {i+1}\n')
            print(f'Source Text: {doc.page_content}')
            print(f'Document Name: {doc.metadata["source"]}')
            print(f'Page Number: {doc.metadata["page"]}\n')
            print('='* 60)
        
        print(f"Time to retrieve response: {end - start}")
        
        if len(args.input) > 0:
            break
        print('='* 80)
        query = input('Enter the query: ')
