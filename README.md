# postgresql_topic_modelling
"Python code that connects to a PostgreSQL database, loads book descriptions from database into a pandas data frame, and performs topic modeling.

The file script.py creates a separate table for the new topics and can only be applied to a table where topics have not been assigned yet. This file trainer a pipeline that can be saved for topic modeling.
The file script_update.py contains a script that selects all entires where no topic has been assigned and adds the topic into the table via an UPDATE clause. It requires a pertained pipe that is stored as pipe.py in the wd.
