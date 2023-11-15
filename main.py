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
    if not query:
        query = input('\nEnter the question: ').strip()
    while query != '\\q':
        if query == '\\timing':
            cfg.TIMING = not cfg.TIMING
            print('Timing is {}'.format('on' if cfg.TIMING else 'off'))
            query = input('\nEnter the query: ')
            continue

        start = timeit.default_timer()
        response = dbqa({'query': query})
        end = timeit.default_timer()
    
        # Process source documents
        source_docs = response['source_documents'] if 'source_documents' in response else []
        for i, doc in enumerate(source_docs):
            print('')
            print('='* 50)
            print(f'\nSource Document {i+1}\n')
            print(f'Source Text: {doc.page_content}')
            if 'source' in doc.metadata:
                print(f'Document Name: {doc.metadata["source"]}')
            if 'page' in doc.metadata:
                print(f'Page Number: {doc.metadata["page"]}\n')
        
        print('='*50)
        print(f'\nQuestion: {query}\n')
        print(f'\nAnswer: {response["result"]}\n')
        if cfg.TIMING:
            print('='*20)
            print(f"Time to retrieve response: {end - start}")
        
        if args.input:
            break
        print('='* 80)
        query = input('\nEnter the query: ')
