from ultralytics import YOLO
#we imported all this lime from the documentation
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='/home/golu/PycharmProjects/weatherImageClassificationYOLOv8/weather/weatherDataset',
            epochs=20, imgsz=64)
# when we see the run and  adn we see the train and val table's value decreaseing with mre epoch and the value of
# accuracy is increasing then we are having a good dataset
# model are stored in weights file

# so while we train after each epoch we have a model which is trained till now, so our last epoch is the best trained
# model as it uses all the data being learnt before adn uses it ,so our last epoch is the best model
#
# which is stored in last_pt in weights but if we see the table we can see that at some time in between we have a
# epoch with better eaccuracyt so its better to use that model inst ead of the last epoch model , so that is wwhy we
# have the best_pt folder adn there we have the best adn highest accuracy model and we can use that
#


# a model chosen after the last epoch can be a good model ,as this as the most maunt of data, or we can choose the
# model with the highest accuracy, so we can choose differnt option based on the use case and the project
