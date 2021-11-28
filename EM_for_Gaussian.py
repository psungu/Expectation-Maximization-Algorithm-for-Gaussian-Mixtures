import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# Following k_means class is taken from: https://github.com/psungu/Cmpe462-Project3

class k_means:

    def __init__(self, data, k, max_iter):
        self.k = k #number of clusters
        self.max_iter = max_iter #number of iterations
        self.data = data #dataset
        
    def fit(self):
        
        #setting initial centroids
        #getting k random integers between 0 and number of rows of data set then selecting rows from dataset via using these integers as row indexes
        slct = np.random.randint(len(self.data), size=self.k)
        self.centroids = self.data[slct]

        #iterate max_iter times which is stopping condition
        for i in range(self.max_iter):
            #declaring empty dictionary to keep clusters
            self.classifications = {}
            #initializing k dictionary elements as empty lists to keep k clusters
            for i in range(self.k):
                self.classifications[i] = []
            
            #for each data point in data set,
            #calculating Euclidean distance between data point and centroids then assigning data point to closest centroid
            #recomputing centroid of each cluster via calculating mean of data points in the clusters
            for point in self.data:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(point)
                
            for classification in self.classifications:
                self.centroids[classification] = np.mean(self.classifications[classification],axis=0)




class GaussianMixtureModel:

    def __init__(self, data, k=3):

        """k is number of mixtures, since in our task 3 is given, set 3 as a default value,
        keep the dimension of the given data"""

        self.x, self.y = data.shape
        self.data = data.copy()
        self.k = k
        self.likelihood_list = []
        self.iteration_count = 0
        
    def __initialize(self, kMeans_initialization=True):
        
        """Initialize mixture means randomly, covariance matrix as identity matrix,
        and equal probability of phi. Prepare the result probability weights as weight"""

        model = k_means(self.data,3,5)
        model.fit()

        if(kMeans_initialization):
            self.mean = np.asmatrix(model.centroids)
        else:
            self.mean = np.asmatrix(np.random.random((self.k, self.y)))

        self.sigma = np.array([np.asmatrix(np.identity(self.y)) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k
        self.weight = np.asmatrix(np.empty((self.x, self.k)))
    
    def __loglikelihood(self):

        """Calculate loglikelihood"""

        likelihood = 0

        for i in range(self.k):

            likelihood += stats.multivariate_normal.pdf(self.data, self.mean[i,:].A1, self.sigma[i,:])*self.phi[i]

        loglikelihood = np.log(likelihood).sum()

        return loglikelihood


    def __E_step(self):

        """Calculate the probability that it belongs to distribution/cluster k_i"""

        sum_of_dist = 0

        for i in range(self.k):

            gamma = stats.multivariate_normal.pdf(self.data, self.mean[i,:].A1, self.sigma[i,:]) * self.phi[i]

            sum_of_dist += gamma

            self.weight[:,i] = gamma[:,None]

        self.weight /= sum_of_dist[:,None]


    
    def __M_step(self):

        """Update phi, mean, and covariance matrix"""

        for j in range(self.k):
            sum_of_weights = self.weight[:, j].sum()
            self.phi[j] = 1/self.x * sum_of_weights
            mu_k_new = np.zeros((self.y,1))
            sigma_k_new = np.zeros((self.y, self.y))
            mu_k_new += (self.data.T @ self.weight[:, j])
            for i in range(self.x):
                sigma_k_new += self.weight[i, j] * np.dot((self.data[i, :] - self.mean[j, :]).T, (self.data[i, :] - self.mean[j, :]))
            self.mean[j] = mu_k_new.reshape(1,2) / sum_of_weights
            self.sigma[j] = sigma_k_new / sum_of_weights


    def fit(self, threshold=0.0001):

        """Call the above private functions until the difference between likelihood and previous likelihood exceed the given threshold"""

        self.__initialize()

        likelihood = 1
        previous_likelihood = 0

        while np.subtract(likelihood, previous_likelihood).all() > threshold:

            previous_likelihood = self.__loglikelihood()

            self.__E_step()
            self.__M_step()

            likelihood = self.__loglikelihood()
            self.likelihood_list.append(likelihood)
            self.iteration_count+=1

        print(f'Iteration converges at likelihood is {likelihood}')

    def plot(self):

        max_index_col = np.argmax(self.weight, axis=1)
        dataset = pd.DataFrame({'X': self.data[:, 0], 'Y': self.data[:, 1]})
        df = pd.DataFrame(dataset)
        df['Cluster'] = max_index_col
        df.columns = ['X', 'Y', 'Cluster']
        color=['red','blue','green']
        for k in range(0,len(color)):
            result_data = df[df["Cluster"]==k]
            plt.scatter(result_data["X"], result_data["Y"], c = color[k])
        
        plt.title('Cluster assignments')
        plt.show()

        # Following script for the report, log likelihood values for each iteration

        x = range(0, len(self.likelihood_list))
        y = self.likelihood_list
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(x, y, color='red', marker='+')
        ax.title.set_text('Log likelihood in each iteration')
        ax.plot(self.likelihood_list)
        plt.show()



def __main__():

    data=np.load('./dataset.npy')
    #plt.hist(data)
    #plt.show()
    gmm = GaussianMixtureModel(data)
    gmm.fit()
    print(f'Convergence terminated at {gmm.iteration_count}-th iteration \n')
    print(f'Estimated means: \n {gmm.mean} \n')
    print(f'Estimated covariance matrix: \n {gmm.sigma} \n')
    print(f'Estimated phis: {gmm.phi}')
    gmm.plot()

__main__()




