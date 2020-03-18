import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import  numpy as np

#LOAD DATA:
cancer = datasets.load_breast_cancer()
cancer_features = cancer.data  # All of the features / type npArray
cancer_target = cancer.target   # All of the labels / type npArray


#SHOW DATASET:
cancer_target_list = cancer.target.tolist()
for i in range(len(cancer_target_list)):
    if cancer_target_list[i] == 0:
        cancer_target_list[i] = "malignant"
    
    if cancer_target_list[i] == 1:
        cancer_target_list[i] = "benign"

df = pd.DataFrame(cancer_features, columns=cancer.feature_names)
df.insert(0, "diagnosis", cancer_target_list)
print("--------------------------------------------")
print(df)
print("--------------------------------------------")


#PLOT DATA

#Create color maps to plot colored points
cmap_dark = ListedColormap(['blue', 'red'])

fig1, axs = plt.subplots(3, 2, num="Cancer-Data", tight_layout=True, figsize=(6, 6))
axs[0, 0].scatter(cancer_features[:, 0], cancer_features[:, 1],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[0, 0].set(xLabel="radius_mean", yLabel="texture_mean")
axs[0, 1].scatter(cancer_features[:, 0], cancer_features[:, 2],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[0, 1].set(xLabel="radius_mean", yLabel="perimeter_mean")
axs[1, 0].scatter(cancer_features[:, 0], cancer_features[:, 3],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[1, 0].set(xLabel="radius_mean", yLabel="area_mean")
axs[1, 1].scatter(cancer_features[:, 1], cancer_features[:, 2],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[1, 1].set(xLabel="texture_mean", yLabel="perimeter_mean")
axs[2, 0].scatter(cancer_features[:, 1], cancer_features[:, 3],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[2, 0].set(xLabel="texture_mean", yLabel="area_mean")
axs[2, 1].scatter(cancer_features[:, 2], cancer_features[:, 3],
                c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs[2, 1].set(xLabel="area_mean", yLabel="area_mean")

plt.show()



#MODEL TRAINING:

#training the model with only the first two features, to show the Hyper Plane!
cancer_features = cancer_features[:, 0:2]
#split train/test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(cancer_features, cancer_target, test_size=0.2)
#find out the best model params
best_acc = 0
for kernel in ['linear', 'rbf', 'sigmoid']:  # poly kernel isnt useful here
    for c in [0.1,0.5,1,3,100,500,800]:
        model = svm.SVC(kernel=kernel, C=c)

        #train the model
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        print("Model :" + kernel +" "+ str(c)+ " acc: "+ str(acc))
        if acc > best_acc:
            best_acc = acc
            best_model = model

model = best_model
#show best model params
print("--------------------------------------------")
print("best model params: \n" + str(model.get_params()))
print("aaccuracy: " + str(best_acc))






#PLOT BEST MODEL AND ORIGINAL

fig1, (axs1, axs2) = plt.subplots(1, 2, num="Predicition", tight_layout=True, figsize=(10,5))


#Original data
axs1.scatter(cancer_features[:, 0], cancer_features[:, 1],c=cancer_target, cmap=cmap_dark, edgecolor='k', s=20)
axs1.set_title("Orignial")


#Model
axs2.scatter(cancer_features[:, 0], cancer_features[:,1], c=cancer_target, s=20, cmap=cmap_dark)
#plot the decision function
axs2 = plt.gca()
xlim = axs2.get_xlim()
ylim = axs2.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
axs2.contour(XX, YY, Z, colors='k',levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# plot support vectors
axs2.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
axs2.set_title("Prediction")

plt.show()
