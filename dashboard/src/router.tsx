import { createBrowserRouter } from "react-router";

import { App } from "./App";
import { Overview } from "./routes/Overview";
import { ReplayReconstruct } from "./routes/ReplayReconstruct";
import { ReviewDetail } from "./routes/ReviewDetail";
import { Reviews } from "./routes/Reviews";
import { SetupGitHubApp } from "./routes/SetupGitHubApp";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: App,
    children: [
      { index: true, Component: Overview },
      { path: "reviews", Component: Reviews },
      { path: "reviews/:reviewId", Component: ReviewDetail },
      { path: "reviews/:reviewId/replay", Component: ReplayReconstruct },
      { path: "setup", Component: SetupGitHubApp },
    ],
  },
]);
