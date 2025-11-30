# Tracker module with centroid fallback and hooks for ByteTrack/DeepSort integration.
import numpy as np

class SimpleTracker:
    def __init__(self, max_lost=5):
        self.nextObjectID = 0
        self.objects = {}
        self.lost = {}
        self.max_lost = max_lost

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.lost[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.lost[objectID]

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)
            return self.objects

        dets = np.array(detections)
        if len(self.objects) == 0:
            for d in dets:
                self.register(tuple(d))
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = np.array([self.objects[i] for i in objectIDs])
        D = np.linalg.norm(objectCentroids[:, None, :] - dets[None, :, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assignedRows = set()
        assignedCols = set()
        for r, c in zip(rows, cols):
            if r in assignedRows or c in assignedCols:
                continue
            oid = objectIDs[r]
            self.objects[oid] = tuple(dets[c])
            self.lost[oid] = 0
            assignedRows.add(r)
            assignedCols.add(c)

        for c in range(dets.shape[0]):
            if c not in assignedCols:
                self.register(tuple(dets[c]))

        for r in range(objectCentroids.shape[0]):
            if r not in assignedRows:
                oid = objectIDs[r]
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)

        return self.objects

# ByteTrack / RAFT wrappers (placeholders) - try to import external libs and provide graceful fallback
try:
    # Example: check for a ByteTrack python interface
    import bytetrack  # likely not installed; this is illustrative
    HAS_BYTETRACK = True
except Exception:
    HAS_BYTETRACK = False

class ByteTrackWrapper:
    def __init__(self, cfg=None):
        if not HAS_BYTETRACK:
            raise RuntimeError('ByteTrack not installed. Install ByteTrack or use centroid tracker.')
        # Implement initialization of ByteTrack here

    def update(self, detections, frame=None):
        # detections: list of bboxes [x1,y1,x2,y2,score,class]
        # returns dict id -> centroid
        raise NotImplementedError('ByteTrack wrapper not implemented in this template.')
