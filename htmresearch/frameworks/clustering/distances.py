import numpy as np



def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between SDRs x1 and x2.

  :param x1:   (array) binary vector or a set of active indices
  :param x2:  (array) binary vector or a set of active indices
  :param size: (int)   length of binary vectors

  :return percentOverlap: (float) percentage overlap between x1 and x2
  """
  if type(x1) is np.ndarray:
    nonZeroX1 = float(np.count_nonzero(x1))
    nonZeroX2 = float(np.count_nonzero(x2))
    minX1X2 = min(nonZeroX1, nonZeroX2)
    percentOverlap = 0
    if minX1X2 > 0:
      percentOverlap = float(np.dot(x1.T, x2)) / np.sqrt(nonZeroX1 * nonZeroX2)
  else:
    nonZeroX1 = len(x1)
    nonZeroX2 = len(x2)
    minX1X2 = min(nonZeroX1, nonZeroX2)
    percentOverlap = 0
    if minX1X2 > 0:
      overlap = float(len(set(x1) & set(x2)))
      percentOverlap = overlap / np.sqrt(nonZeroX1 * nonZeroX2)

  return percentOverlap



def clusterDist(c1, c2):
  """
  symmetric distance between two clusters

  :param c1: (np.array) cluster 1
  :param c2: (np.array) cluster 2
  :return: distance between 2 clusters
  """
  d12 = clusterDistDirected(c1, c2)
  d21 = clusterDistDirected(c2, c1)
  return np.mean([d12, d21])



def overlapDistance(sdr1, sdr2):
  return 1 - percentOverlap(sdr1, sdr2)



def clusterDistDirected(c1, c2):
  """
  Directed distance from cluster 1 to cluster 2

  :param c1: (np.array) cluster 1
  :param c2: (np.array) cluster 2
  :return: distance between 2 clusters
  """
  minDists = []
  for sdr1 in c1:
    d = []
    # ignore SDRs with zero active bits
    if np.sum(sdr1) == 0:
      continue

    for sdr2 in c2:
      d.append(1 - percentOverlap(sdr1, sdr2))
    minDists.append(min(d))
  return np.mean(minDists)



def kernel_dist(kernel):
  return lambda x, y: kernel(x, x) - 2 * kernel(x, y) + kernel(y, y)



def pointsToSDRs(points):
  return [p.getValue() for p in points]



def interClusterDistances(clusters, newCluster):
  interClusterDist = {}
  if len(clusters) > 0:
    for c1 in clusters:
      for c2 in clusters:
        name = "c%s-c%s" % (c1.getId(), c2.getId())
        interClusterDist[name] = clusterDist(pointsToSDRs(c1.getPoints()),
                                             pointsToSDRs(c2.getPoints()))
      if len(newCluster.getPoints()) > 0:
        name = "c%s-new%s" % (c1.getId(), newCluster.getId())
        interClusterDist[name] = clusterDist(
          pointsToSDRs(c1.getPoints()), pointsToSDRs(newCluster.getPoints()))
  return interClusterDist



def computeClusterDistances(cluster, clusters):
  """
  Compute distance between a cluster and each cluster in a list of clusters 
  :param cluster: (Cluster) cluster to compare to other clusters
  :param clusters: (list of Cluster objects) 
  :return: ordered list of clusters and their distances 
  """
  dists = []
  for c in clusters:
    d = clusterDist(pointsToSDRs(c.getPoints()),
                    pointsToSDRs(cluster.getPoints()))
    dists.append((d, c))

  clusterDists = sorted(dists, key=lambda x: (x[0], x[1]._lastUpdated))

  return clusterDists
