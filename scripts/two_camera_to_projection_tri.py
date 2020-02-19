F, mask = findFundamentalMat(pts1, pts2)

E = np.dot(K2.T, np.dot(F, K1))  # K2'*F*K1
_, R, t, _ = cv.recoverPose(E, pts1[mask], pts2[mask], self.K)


P1 = np.column_stack([np.eye(3), np.zeros(3)]) # webcam is origin
P2 = np.hstack((R, t))

pts1_norm = cv.undistortPoints(pts1, cameraMatrix = K1, distCoeffs = dist1)
pts2_norm = cv.undistortPoints(pts2, cameraMatrix = K2, distCoeffs = dist1)

points_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)

points_3d = cv.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)


