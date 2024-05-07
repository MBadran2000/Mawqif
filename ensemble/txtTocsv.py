import csv,os

def convert(txt_file = 'input.txt',csv_file= 'output.csv'):
    # print(txt_file,csv_file)
    # Open the input and output files
    with open(txt_file, 'r', encoding='utf-8') as input_file, open(csv_file, 'w', newline='', encoding='utf-8') as output_file:
        # Create a CSV writer
        csv_writer = csv.writer(output_file)

        # Write header
        csv_writer.writerow(['Index', 'Category', 'Text', 'Label'])

        # Read and process each line from the input file
        for line in input_file:
            # Split the line by tab character
            parts = line.strip().split('\t')
            
            # Write the parts into the CSV file
            csv_writer.writerow(parts)

    print("Conversion completed successfully.")

for i in os.listdir('results'):
    if '.csv' in i or 'ensembles' in i:
        continue
    print(i)
    convert("results/"+i+"/test_pred_File.txt","results/"+i+".csv")
    # break

convert("results/V101.1/test_gt_file.txt","results/gt.csv")
