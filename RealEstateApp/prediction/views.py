
from django.shortcuts import render
import pandas as pd
import joblib

#load the machine learning model
# reloadModel = joblib.load('./models/pipeline.pkl')
Model = joblib.load('./models/pipeline.pkl')

# Create your views here.
def prediction(request):
    temp={}
    context= {'temp':temp}
    return render(request, 'base.html',context)

def predict(request):
    temp=dict()
    if request.method == 'POST':
            
        temp=dict()
        temp['City'] = [request.POST.get('City')]
        temp['Address'] = [request.POST.get('Address')]
        temp['Bedroom'] = [request.POST.get('Bedroom')]
        temp['Bathroom'] = [request.POST.get('Bathroom')]
        temp['Floors'] = [request.POST.get('Floors')]
        temp['Parking'] = [request.POST.get('Parking')]
        temp['Year'] = [request.POST.get('Year')]
        temp['Face'] = [request.POST.get('Face')]
        temp['Area'] = [request.POST.get('Area')]
        temp['Road_Width'] = [request.POST.get('Road_Width')]
        temp['Road_Type'] = [request.POST.get('Road_Type')]
        temp['Build_Area'] = [request.POST.get('Build_Area')]
        temp['Amenities'] = [request.POST.get('Amenities')]

    data_df = pd.DataFrame(temp)
    print(data_df)
    ans= int(Model.predict(data_df))
    print(ans)
    # ans= int(reloadModel.predict(data_df))

    context={'scoreval':ans,'temp':temp}
    return render(request,'predict.html',context)

