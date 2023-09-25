# postgresql_topic_modelling
"This Python script connects to a PostgreSQL database using Psycopg2, fetches book descriptions from the database, and organizes them into a Pandas DataFrame. It then performs topic modeling on this data.

script.py: This script creates a dedicated table for storing the new topics. It should only be used with a table where topics have not yet been assigned. This script is responsible for training a pipeline that can later be saved for topic modeling.
script_update.py: In this script, we select all entries where no topic has been assigned and update the table by adding the topics using an UPDATE clause. This script relies on a pre-trained pipeline stored as pipe.py in the working directory (wd) for its operation.
