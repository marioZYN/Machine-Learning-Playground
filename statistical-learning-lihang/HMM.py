class HMM():

    def forward_backward(self, y):
        # set up
        if y.ndim == 2:
            y = y[np.newaxis, ...]

        nB, nT = y.shape[:2]

        posterior = np.zeros((nB, nT, self.K))
        forward = np.zeros((nB, nT + 1, self.K))
        backward = np.zeros((nB, nT + 1, self.K))

        # forward pass
        forward[:, 0, :] = 1.0 / self.K
        for t in range(nT):
            tmp = np.multiply(
                np.matmul(forward[:, t, :], self.P),
                y[:, t]
            )
            # normalize
            forward[:, t + 1, :] = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

        # backward pass
        backward[:, -1, :] = 1.0 / self.K
        for t in range(nT, 0, -1):
            # TODO[marcel]: double check whether y[:,t-1] should be y[:,t]
            tmp = np.matmul(self.P, (y[:, t - 1] * backward[:, t, :]).T).T
            # normalize
            backward[:, t - 1, :] = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

        # remove initial/final probabilities and squeeze for non-batched tests
        forward = np.squeeze(forward[:, 1:, :])
        backward = np.squeeze(backward[:, :-1, :])

        # TODO[marcel]: posterior missing initial probabilities
        # combine and normalize
        posterior = np.array(forward) * np.array(backward)
        # [:,None] expands sum to be correct size
        posterior = posterior / np.sum(posterior, axis=-1)[..., np.newaxis]

        # squeeze for non-batched tests
        return posterior, forward, backward

    def viterbi_forward(self, scores):
        tmpMat = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                tmpMat[i, j] = scores[i] + self.logP[i, j]
        return tmpMat

    def viterbi_decode(self, y):
        nT = y.shape[0]

        pathStates = np.zeros((nT, self.K), dtype=np.int)
        pathScores = np.zeros((nT, self.K))

        # initialize
        pathScores[0] = self.logp0 + np.log(y[0])

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])
            # the inferred state
            pathStates[t + 1] = np.argmax(tmpMat, 0)
            pathScores[t + 1] = np.max(tmpMat, 0) + np.log(yy)

        # now backtrack viterbi to find states
        s = np.zeros(nT, dtype=np.int)
        s[-1] = np.argmax(pathScores[-1])
        for t in range(nT - 1, 0, -1):
            s[t - 1] = pathStates[t, s[t]]

        return s, pathScores