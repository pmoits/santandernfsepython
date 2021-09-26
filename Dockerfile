FROM mcr.microsoft.com/playwright:bionic
WORKDIR .
COPY package.json package*.json ./nfse-python/
ENV PLAYWRIGHT_BROWSERS_PATH=0
RUN npm ci
COPY . .
CMD ["node", "src/main.js"]