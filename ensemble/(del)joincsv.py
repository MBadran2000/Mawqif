import csv

def read_csv(filename):
    data = {}
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index = row['Index']
            category = row['Category']
            text = row['Text']
            label = row['Label']
            key = (index, category, text)
            data[key] = label
    return data

def write_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Category', 'Text', 'Label1', 'Label2'])
        for key, labels in data.items():
            index, category, text = key
            label1, label2 = labels
            writer.writerow([index, category, text, label1, label2])

def join_csv(csv1, csv2, output_csv):
    data1 = read_csv(csv1)
    data2 = read_csv(csv2)
    combined_data = {}
    for key, label1 in data1.items():
        if key in data2:
            label2 = data2[key]
            combined_data[key] = (label1, label2)
    write_csv(combined_data, output_csv)

# Replace 'file1.csv', 'file2.csv', and 'output.csv' with the appropriate file names
# join_csv('results/V101.0.csv', 'results/V101.1.csv', 'results/combined.csv')


for i in os.listdir('results'):
    if '.csv' in i:
        print(i)
        convert("results/"+i+"/test_pred_File.txt","results/"+i+".csv")
    # break