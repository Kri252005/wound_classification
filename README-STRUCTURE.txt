WOUNDSCOPE - SIMPLE PROJECT STRUCTURE

src/main.tsx
    React entry point. Starts the application.

src/App.tsx
    Main page. Arranges the major sections.

src/components/Header.tsx
    Navigation bar and mobile menu.

src/components/Hero.tsx
    Landing/overview section.

src/components/Classifier.tsx
    File upload, image preview, classification state and user actions.

src/components/ResultPanel.tsx
    Displays the prediction returned by the backend.

src/components/ConfidenceBars.tsx
    Displays class probabilities.

src/components/Methodology.tsx
    Displays the ML pipeline explanation.

src/components/Research.tsx
    Displays research metrics.

src/services/api.ts
    Sends the image to FastAPI at POST /predict.

src/types/prediction.ts
    Defines the TypeScript structure of the backend response.

DATA FLOW

User
  -> Classifier.tsx
  -> services/api.ts
  -> FastAPI /predict
  -> ML model
  -> PredictionResponse
  -> Classifier.tsx
  -> ResultPanel.tsx
  -> ConfidenceBars.tsx

IMPORTANT

The Research.tsx values are copied from the existing UI shown in the
original project. Replace them with your actual final experimental metrics
if they are placeholders.
