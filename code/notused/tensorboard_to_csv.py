from tensorboard.backend.event_processing import event_file_loader
import csv

# Path to the event file
event_file_path = "C:/Users/uqjwil54/Documents/Projects/DBBD/2024_05_10/balanced/models/2024_05_13/events.out.tfevents.1715309966.BIOL-42Y5TC3.8800.0"
# Create an EventFileLoader
loader = event_file_loader.EventFileLoader(event_file_path)

# Load all events from the event file
events = list(loader.Load())

# Create empty lists to store values
wall_times = []
steps = []
tags = []

results = {"wall_time": [], "step": [], "AP:0.5-0.95": [], "AP:0.5": [], "AP:0.75": [], "AR:0.5-0.95": [], "F1": []}
# Extract wall_time and step from each event and append them to the lists
for event in events:
    for value in event.summary.value:
        if value.tag in ["AP:0.5"]:
            results["wall_time"].append(event.wall_time)
            results["step"].append(event.step)
        if value.tag in ["AP:0.5-0.95", "AP:0.5", "AP:0.75", "AR:0.5-0.95", "F1"]:
            results[value.tag].append(value.tensor.float_val)

# Save results to a CSV file
output_csv_file = 'output.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ["wall_time", "step", "AP:0.5-0.95", "AP:0.5", "AP:0.75", "AR:0.5-0.95", "F1"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(results["wall_time"])):
        # print(i)
        row = {key: results[key][i] for key in fieldnames}
        writer.writerow(row)

print(f"CSV file saved to: {output_csv_file}")



