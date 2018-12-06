import csv
import pdb
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

dataset_dictionary = {}
header_index = {}

#datatype = ["int", "int", "str", "str", "str", "int", "int", "str", "str", "str", "int", "int", "bool", "bool", "str", "str", "bool", "int", "str", "str"]

with open('data/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',' )
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')

            for header in range(0, len(row)):
                dataset_dictionary[row[header]] = []
                header_index[header] = row[header]
            line_count += 1
        else:
            ##lets skip points with salary range at 0
            if row[0] == "0" or row[1] == "0" or row[0] == "" or row[1] == "":
                continue
            
            for i in range(0, len(row)):
                dataset_dictionary[ header_index[i] ].append(row[i])
            
            line_count += 1


##now lets concatenate all the words we care about for a particular job posting
#job title, seniority, address, skills, roles, industries, stage
## then I'll make words for "is_public, is_acquired, and remote_ok" based on T/F values
for i in range(len(dataset_dictionary["address"])):
    dataset_dictionary["address"][i] = dataset_dictionary["address"][i].replace(" ", "_")


dataset_dictionary["avg_salary"] = []
for i in range(len(dataset_dictionary["min_salary"])):
    avg = ( float(dataset_dictionary["min_salary"][i]) + float(dataset_dictionary["max_salary"][i] ) ) / 2
    dataset_dictionary["avg_salary"].append(avg)

dataset_processed = zip( dataset_dictionary["job_title"], dataset_dictionary["seniority"], dataset_dictionary["address"], dataset_dictionary["skills"], dataset_dictionary["roles"], dataset_dictionary["industries"], dataset_dictionary["stage"] )

# ('Linux C Developer', 'regular', 'Salt Lake City_UT_USA', 'artificial_intelligence,deep_learning,machine_learning,linux,c,embedded', 'developer', 'consumer_electronics,aerospace,software,drones', '')


dataset_merged = []

for record in dataset_processed:
    ##got a tuple, now iterate through all its values
    new_list  = []
    for t in record:
        tlist = t.split(",")
        tlist = [val for val in tlist if val != '']
        
        new_list.extend(tlist)
    dataset_merged.append(new_list)

for i in range(len(dataset_merged)):
    job_desc = dataset_merged[i][0]
    job_desc = job_desc.replace("/", " ")
    job_desc = job_desc.replace("(", " ")
    job_desc = job_desc.replace(")", " ")
    job_desc = job_desc.lower()
    
    job_desc = re.sub(r'back[\s-]end', "backend", job_desc)    
    job_desc = re.sub(r'front[\s-]end', "frontend", job_desc)    
    ##fix front end and backend
    cleaned_desc = job_desc.split(" ")
    del dataset_merged[i][0]
    new_list = cleaned_desc + dataset_merged[i]
    dataset_merged[i] = new_list

for i in range(len(dataset_merged)):
    pub = dataset_dictionary["is_public"][i] 
    acq = dataset_dictionary["is_acquired"][i]
    remote = dataset_dictionary["remote_ok"][i]

    if pub == "True":
        dataset_merged[i].append("is_pub")
    elif pub == "False":
        dataset_merged[i].append("isnt_pub")

    if acq == "True":
        dataset_merged[i].append("is_acq")
    elif acq == "False":
        dataset_merged[i].append("isnt_acq")

    if remote == "True":
        dataset_merged[i].append("is_remote")
    elif remote == "False":
        dataset_merged[i].append("isnt_remote")

dataset_strings = []
for i in range(len(dataset_merged)):
    dataset_strings.append( " ".join(dataset_merged[i]) )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset_strings)

#print(vectorizer.get_feature_names() )
print(X.shape)


##train test split 
salary_options = ["avg_salary", "min_salary", "max_salary"]
salary_model_data = {}

##convert min and max salaries to floats
for sal_in in range(len(dataset_dictionary["min_salary"])):
    dataset_dictionary["min_salary"][sal_in] = float( dataset_dictionary["min_salary"][sal_in] )
    dataset_dictionary["max_salary"][sal_in] = float( dataset_dictionary["max_salary"][sal_in] )


for el in salary_options:
    X_train, X_test, y_train, y_test = train_test_split( X, dataset_dictionary[el] , test_size=250)
    salary_model_data[el] = {}
    salary_model_data[el]["X_train"] = X_train
    salary_model_data[el]["X_test"] = X_test
    salary_model_data[el]["y_train"] = y_train
    salary_model_data[el]["y_test"] = y_test

    salary_model_data[el]["regressor"] = LinearRegression(fit_intercept=True)

for el in salary_options:
    salary_model_data[el]["regressor"].fit(  X=salary_model_data[el]["X_train"], \
            y=salary_model_data[el]["y_train"] )

##Calculate how good the model is
for el in salary_options:
    #mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
    y_pred = salary_model_data[el]["regressor"].predict( salary_model_data[el]["X_test"] )
    y_true = salary_model_data[el]["y_test"]

    salary_model_data[el]["MSE"] = mean_squared_error(y_true, y_pred)
    salary_model_data[el]["MAE"] = mean_absolute_error(y_true, y_pred)
    salary_model_data[el]["MEDAE"] = median_absolute_error(y_true, y_pred)
    salary_model_data[el]["r2"] = r2_score(y_true, y_pred)

metric_names = ["MSE", "MAE", "MEDAE", "r2"]
for el in salary_options:
    print("Summary data for " + str(el) + " model ")
    for m in metric_names:
        print(m + ":")
        print(salary_model_data[el][m])


print("Done reading in.")


