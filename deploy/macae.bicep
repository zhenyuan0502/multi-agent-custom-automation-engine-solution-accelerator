@description('Location for all resources.')
param location string = 'EastUS2' //Fixed for model availability, change back to resourceGroup().location

@description('Location for OpenAI resources.')
param azureOpenAILocation string = 'EastUS' //Fixed for model availability

@description('A prefix to add to the start of all resource names. Note: A "unique" suffix will also be added')
param prefix string = 'macae'

@description('The container image name and tag to deploy to the backend container app, if this is not set the container app just default to an empty image')
param backendContainerImageNameTag  string = ''

@description('The container image name and tag for the frontend application, if this is not set the container app just default to an empty image')
param frontendContainerImageNameTag string = ''

@secure()
@description('The visitor code/password that must be provided to access the container app. If left as-is, a new password will be generated and output. If this is set to empty a new code will be generated on each restart.')
param visitorPassword string = base64(newGuid())

@description('Tags to apply to all deployed resources')
param tags object = {}

@description('The size of the resources to deploy, defaults to a mini size')
param resourceSize {
  gpt4oCapacity: int
  cosmosThroughput: int
  containerAppSize: {
    cpu: string
    memory: string
    minReplicas: int
    maxReplicas: int
  }
} = {
  gpt4oCapacity: 15
  cosmosThroughput: 400
  containerAppSize: {
    cpu: '1.0'
    memory: '2.0Gi'
    minReplicas: 0
    maxReplicas: 1
  }
}

var uniqueNameFormat = '${prefix}-{0}-${uniqueString(resourceGroup().id, prefix)}'
var uniqueShortNameFormat = '${toLower(prefix)}{0}${uniqueString(resourceGroup().id, prefix)}'
var aoaiApiVersion = '2024-08-01-preview'
var emptyContainerImage = 'alpine:latest'
param frontendSiteName string = '${prefix}-frontend-${uniqueString(resourceGroup().id)}'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: format(uniqueNameFormat, 'logs')
  location: location
  tags: tags
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02-preview' = {
  name: format(uniqueNameFormat, 'appins')
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource openai 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: format(uniqueNameFormat, 'openai')
  location: azureOpenAILocation
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: format(uniqueNameFormat, 'openai')
  }
  resource gpt4o 'deployments' = {
    name: 'gpt-4o'
    sku: {
      name: 'GlobalStandard'
      capacity: resourceSize.gpt4oCapacity
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: 'gpt-4o'
        version: '2024-08-06'
      }
      versionUpgradeOption: 'NoAutoUpgrade'
    }
  }
}

resource aoaiUserRoleDefinition 'Microsoft.Authorization/roleDefinitions@2022-05-01-preview' existing = {
  name: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' //'Cognitive Services OpenAI User'
}

resource acaAoaiRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerApp.id, openai.id, aoaiUserRoleDefinition.id)
  scope: openai
  properties: {
    principalId: containerApp.identity.principalId
    roleDefinitionId: aoaiUserRoleDefinition.id
    principalType: 'ServicePrincipal'
  }
}

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: format(uniqueNameFormat, 'cosmos')
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    enableFreeTier: false
    locations: [
      {
        failoverPriority: 0
        locationName: location
      }
    ]
  }

  resource contributorRoleDefinition 'sqlRoleDefinitions' existing = {
    name: '00000000-0000-0000-0000-000000000002'
  }

  resource autogenDb 'sqlDatabases' = {
    name: 'autogen'
    properties: {
      resource: {
        id: 'autogen'
        createMode: 'Default'
      }
      options: {
        throughput: resourceSize.cosmosThroughput
      }
    }

    resource memoryContainer 'containers' = {
      name: 'memory'
      properties: {
        resource: {
          id: 'memory'
          partitionKey: {
            kind: 'Hash'
            version: 2
            paths: [
              '/session_id'
            ]
          }
        }
      }
    }
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: format(uniqueShortNameFormat, 'acr')
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true  // Add this line
  }
}

resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-07-31-preview' = {
  name: format(uniqueNameFormat, 'containerapp-pull')
  location: location
}

resource acrPullDefinition 'Microsoft.Authorization/roleDefinitions@2022-05-01-preview' existing = {
  name: '7f951dda-4ed3-4680-a7ca-43fe172d538d' //'AcrPull'
}

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, pullIdentity.id, acrPullDefinition.id)
  properties: {
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullDefinition.id
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: format(uniqueNameFormat, 'containerapp')
  location: location
  tags: tags
  properties: {
    daprAIConnectionString: appInsights.properties.ConnectionString
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
  resource aspireDashboard 'dotNetComponents@2024-02-02-preview' = {
    name: 'aspire-dashboard'
    properties: {
      componentType: 'AspireDashboard'
    }
  }
}

resource acaCosomsRoleAssignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  name: guid(containerApp.id, cosmos::contributorRoleDefinition.id)
  parent: cosmos
  properties: {
    principalId: containerApp.identity.principalId
    roleDefinitionId: cosmos::contributorRoleDefinition.id
    scope: cosmos.id
  }
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-backend'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned,UserAssigned'
    userAssignedIdentities: {
      '${pullIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        targetPort: 8000
        external: true
      }
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: acr.properties.loginServer
          identity: pullIdentity.id
        }
      ]
    }
    template: {
      scale: {
        minReplicas: resourceSize.containerAppSize.minReplicas
        maxReplicas: resourceSize.containerAppSize.maxReplicas
        rules: [
          {
            name: 'http-scaler'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
      containers: [
        {
          name: 'backend'
          image: empty(trim(backendContainerImageNameTag ))
            ? emptyContainerImage
            : '${acr.properties.loginServer}/${backendContainerImageNameTag }'
          resources: {
            cpu: json(resourceSize.containerAppSize.cpu)
            memory: resourceSize.containerAppSize.memory
          }
          env: [
            {
              name: 'COSMOSDB_ENDPOINT'
              value: cosmos.properties.documentEndpoint
            }
            {
              name: 'COSMOSDB_DATABASE'
              value: cosmos::autogenDb.name
            }
            {
              name: 'COSMOSDB_CONTAINER'
              value: cosmos::autogenDb::memoryContainer.name
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openai.properties.endpoint
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: openai::gpt4o.name
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: aoaiApiVersion
            }
            {
              name: 'VISITOR_PASSWORD'
              value: visitorPassword
            }
            {
              name: 'FRONTEND_SITE_NAME'
              value: 'https://${frontendSiteName}.azurewebsites.net'
            }
          ]
        }
      ]
    }
  }
}

resource frontendAppServicePlan 'Microsoft.Web/serverfarms@2021-02-01' = {
  name: '${prefix}-frontend-plan-${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: {
    name: 'P1v2'
    capacity: 1
    tier: 'PremiumV2'
  }
  properties: {
    reserved: true
  }
  kind: 'linux'  // Add this line to support Linux containers
}

resource frontendAppService 'Microsoft.Web/sites@2021-02-01' = {
  name: frontendSiteName
  location: location
  tags: tags
  kind: 'app,linux,container'  // Add this line
  properties: {
    serverFarmId: frontendAppServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion:'DOCKER|nginx:latest'
      appSettings: [
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${acr.properties.loginServer}'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_USERNAME'
          value: acr.listCredentials().username
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_PASSWORD'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'WEBSITES_PORT'
          value: '3000'
        }
        {
          name: 'WEBSITES_CONTAINER_START_TIME_LIMIT'  // Add startup time limit
          value: '1800'  // 30 minutes, adjust as needed
        }
        {
          name: 'BACKEND_API_URL'
          value: 'https://${containerApp.properties.configuration.ingress.fqdn}'
        }
      ]
    }
  }
}

var backendBuildImageTag = 'backend:latest'
var frontendBuildImageTag = 'frontend:latest'

output buildBackendCommand string = 'az acr build -r ${acr.name} -t ${acr.name}.azurecr.io/${backendBuildImageTag} ./macae/backend'
output runBackendCommand string = 'az containerapp update -n ${containerApp.name} -g ${resourceGroup().name} --image ${acr.properties.loginServer}/${backendBuildImageTag}'
output buildFrontendCommand string = 'az acr build -r ${acr.name} -t ${acr.name}.azurecr.io/${frontendBuildImageTag} ./macae/frontend'
output runFrontendCommand string = 'az webapp config container set --name ${frontendAppService.name} --resource-group ${resourceGroup().name} --docker-custom-image-name ${acr.properties.loginServer}/${frontendBuildImageTag} --docker-registry-server-url ${acr.properties.loginServer}'
output cosmosAssignCli string = 'az cosmosdb sql role assignment create --resource-group "${resourceGroup().name}" --account-name "${cosmos.name}" --role-definition-id "${cosmos::contributorRoleDefinition.id}" --scope "${cosmos.id}" --principal-id "fill-in"'
output backendApiUrl string = containerApp.properties.configuration.ingress.fqdn
