import { createBrowserRouter } from "react-router";

import { App } from "./App";
import { Home } from "./routes/Home";
import { ReviewDetail } from "./routes/ReviewDetail";
import { Reviews } from "./routes/Reviews";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: App,
    children: [
      { index: true, Component: Home },
      { path: "reviews", Component: Reviews },
      { path: "reviews/:reviewId", Component: ReviewDetail },
    ],
  },
]);
