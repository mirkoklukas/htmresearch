# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Ideal observer model with same experimental setup as convergence simulation.

Generate objects with ten positions out of a four-by-four grid that have
features randomly selected from a pool of unique features.
"""

import collections
import json
import random

import numpy as np


def generateObjects(numObjects, numFeatures):
  objects = {}
  for i in xrange(numObjects):
    obj = np.zeros((16,), dtype=np.int32)
    obj.fill(-1)
    obj[:10] = np.random.randint(numFeatures, size=10, dtype=np.int32)
    np.random.shuffle(obj)
    objects[i] = obj.reshape((4, 4))
  return objects


def getStartingSpots(objects):
  startingSpots = collections.defaultdict(list)
  for i, obj in objects.iteritems():
    for x in xrange(4):
      for y in xrange(4):
        feat = obj[x, y]
        if feat != -1:
          startingSpots[feat].append((i, (x, y)))
  return startingSpots


def runTrial(objects, startingSpots, numFeatures):
  numObjects = len(objects)

  results = collections.defaultdict(int)
  for targetID in xrange(numObjects):
    targetObject = objects[targetID]

    possibleObjects = None

    possiblePositions = range(16)
    random.shuffle(possiblePositions)
    steps = 0

    for pos in possiblePositions:

      x = pos % 4
      y = pos / 4
      feat = targetObject[x, y]

      if feat == -1:
        continue

      steps += 1
      curPos = (x, y)

      if possibleObjects is None:
        possibleObjects = startingSpots[feat]
      else:
        changeX = x - prevPos[0]
        changeY = y - prevPos[1]

        newPossibleObjects = []
        for objectID, coords in possibleObjects:
          newX = coords[0] + changeX
          newY = coords[1] + changeY
          try:
            expectedFeat = objects[objectID][newX, newY]
          except IndexError:
            continue
          if expectedFeat == feat:
            newPossibleObjects.append((objectID, (newX, newY)))
        possibleObjects = newPossibleObjects

      possibleObjectIDs = set([pair[0] for pair in possibleObjects])
      if len(possibleObjects) == 1:
        assert list(possibleObjectIDs)[0] == targetID
        results[steps] += 1
        break

      prevPos = curPos

    assert len(possibleObjects) > 0

    #if len(possibleObjectIDs) > 1:
    if len(possibleObjects) > 1:
      results[None] += 1

  return results


def runSim(numObjects, numFeatures, numTrials):
  # Map from # sensations to list of number of objects per trial
  results = collections.defaultdict(list)
  for _ in xrange(numTrials):
    objects = generateObjects(numObjects, numFeatures)
    # Built map from a feature to all possible positions
    startingSpots = getStartingSpots(objects)
    trialResults = runTrial(objects, startingSpots, numFeatures)
    for steps, count in trialResults.iteritems():
      results[steps].append(count)

  results = dict(results)

  with open("results/ideal.json", "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  runSim(100, 10, 10)
