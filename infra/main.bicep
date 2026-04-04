// ---------------------------------------------------------------------------
// Qwen3-8B Pipeline — Azure Container Registry (Basic SKU)
//
// Provisions only the registry. The container instance is created on-demand
// by azure_pipeline.sh and torn down after the run to avoid idle charges.
// ---------------------------------------------------------------------------
@description('Location for all resources.')
param location string = resourceGroup().location

@description('ACR name — must be globally unique, 5-50 alphanumeric chars.')
param acrName string = 'qwen3pipelineacr'

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
