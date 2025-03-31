# Neo4j Insight

Master | Build | Cross Version Tests
:---: | :---: | :---:
[![master](https://live.neo4j-build.io/app/rest/builds/buildType:(UserInterface_Neo4jInsight_Test)/statusIcon)](https://live.neo4j-build.io/viewType.html?buildTypeId=UserInterface_Neo4jInsight_Test) | [![build](https://live.neo4j-build.io/app/rest/builds/buildType:(id:UserInterface_Neo4jInsight_Build)/statusIcon)](https://live.neo4j-build.io/viewType.html?buildTypeId=UserInterface_Neo4jInsight_Build) | [![cross version tests](https://live.neo4j-build.io/app/rest/builds/buildType:(UserInterface_Neo4jInsight_CrossVersionTest)/statusIcon)](https://live.neo4j-build.io/viewType.html?buildTypeId=UserInterface_Neo4jInsight_CrossVersionTest)

## Development setup
1. Clone this repo
2. Install yarn globally (not required but recommended): `npm install -g yarn`
3. Add NPM Access Token for @neo4j-bloom libraries, see [setup](/docs/setup.md)
4. Install project dependencies: `yarn`

### Development server
`yarn dev` and point your web browser to `http://localhost:8085`.

### Setting up Neo4j Desktop
The Bloom running locally needs to connect to Neo4j Desktop (where the Neo4j DBMS lives). To do this, start [Neo4j Desktop](https://neo4j.com/download-neo4j-now).

Once this is set up, start a DBMS in Desktop - Desktop will expose this active DBMS to the default port bolt://localhost:7687,and Bloom will use that default port for connecting to the instance.

The login credentials on the Bloom page is the same as the login credentials of the DBMS (note - the default username is `neo4j`, the password can be reset for that DBMS in Desktop)

### Setting up Neo4j Aura
Neo4j Aura is the Neo4j database running in the cloud. In order to test Bloom against Aura, perform the following steps:
1. Log in into [Aura](https://console.neo4j.io/) and create a free database
2. Install the cerificate `neo4j-insight/openssl/bloom-fake.neo4j.io.crt` in your system (on Mac, open keychain access, drag and drop the certificate, double click it and choose "Always trust")
3. In a terminal, go to _/etc/hosts_ file and add `127.0.0.1 bloom-fake.neo4j.io` to a new line
4. Run `yarn startSSL` 
5. Connect via `https://bloom-fake.neo4j.io:8085?connectURL=neo4j%2Bs://{aura_dbms_id}.databases.neo4j.io:7687` where 'aura_dbms_id' can be found in the _Connection URI_ in ur newly created Aura database (note: in Chrome, type `thisisunsafe` in order to bypass the ssl checks).

#### Testing a release against Aura
Perform the same steps as above for installing the certificate and editing the hosts file. Then Bloom can be hosted using serve with the below command:
`npx serve -l tcp://bloom-fake.neo4j.io --ssl-cert={path_to_bloom}/neo4j-insight/openssl/bloom-fake.neo4j.io.crt --ssl-key={path_to_bloom}/neo4j-insight/openssl/bloom-fake.neo4j.io.key assets`

### Setting up Neo4j Enterprise
Please have a look at this [doc](/docs/docker.md) for setup instructions.


### Styling
This repo uses [tailwindcss](https://tailwindcss.com/), with [Neo4j design system](https://github.com/neo4j/neo4j-design) as a extension. All CSS is set in `src/main.css`.

`src/main.css` is processed by PostCss and generate the actual CSS used in the app. Whenever you use a tailwind class name or a change is made in `src/main.css` Vite will process it automatically (`yarn vite:dev`).

[Tailwind CSS IntelliSense](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss) is recommended to get css name auto-hint.

To enable Tailwind CSS IntelliSense with [classnames](https://github.com/JedWatson/classnames), please add the following entry to the VS Code Settings JSON
```
  "tailwindCSS.experimental.classRegex": [
        ["classnames\\(([^)]*)\\)", "'([^']*)'"],
        ["cn\\(([^)]*)\\)", "'([^']*)'"]
    ]
```
`classnames` and `cn` is the imported name,
```
import cn from 'classnames' 
```
or
```
import classnames from 'classnames'
```


### Formatting and linting
- Prettier is used for formatting with [standard](https://github.com/standard/standard) rules and `.prettierrc.js`.
- ESLint is used for linting with [standard](https://github.com/standard/standard) rules and `.eslintrc.json`. ESLint is included in `prettier-standard`.

The lint job will also be performed when committing the change (`git commit`), using husky and lint-staged. This enforces all the change to meet the criteria and makes this process IDE agnostic.

If you need to disable pre-commit hook, please empty `lint-staged` in `package.json` 
```
  "lint-staged": {}    // <- set to empty
```

For Visual Studio Code user, to facilitate your workflow, you can install plugin [prettier-standard](https://marketplace.visualstudio.com/items?itemName=numso.prettier-standard-vscode) and [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint), and add the following settings.

When adding the settings to VSCode, it is advised to add them to the **Workspace settings** per this repo only because other projects are not using this formatting/linting configuration. So we can avoid changing things automatically in other projects like NX, NDL and NVL

```
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "[javascriptreact]": {
    "editor.defaultFormatter": "numso.prettier-standard-vscode"
  },
  "[javascript]": {
    "editor.defaultFormatter": "numso.prettier-standard-vscode"
  }
```
This ensures that whenever a file is saved, it's also automatically formatted and linted with all auto-fixes. You might still need to correct some issues manually.

### e2e Testing with playwright
- `yarn e2e:install` get all the browser driver you need for playwright (you only need to run this for the first time)
- `yarn e2e` run all test under ./playwright/pullrequest folder with chromium
- `yarn e2e:plugin` run all test under ./playwright/plugin folder with chromium
- We are using Movie*.property.dump as test data, you can find the .dump file under [./bloom-test-files](https://s3.console.aws.amazon.com/s3/buckets/bloom-test-files?region=eu-west-2&tab=objects) in S3
- Test artifacts can be found under ./playwright/artifacts/ folder, only after test failed. And every time you launch a new playwright job, it will wipe out the artifacts from last job. Using [trace](https://trace.playwright.dev/) to check test report.
- `npx playwright test /playwright/pullrequest/cardlist.test.ts --project=chromium --config playwright/playwright.config.ts` using this to run single test, you can also install playwright plugin in Visual Studio to help you.
- Please remember to update test coverage map after you change e2e test or developed a new feature: [bloom_feature_list](https://docs.google.com/spreadsheets/d/1f4hxZgwbd0rOXKPaPW8jrh3ta4gIfu4wZ0IPVQZG0xw/edit#gid=2142202886), [Miro_feature_map](https://miro.com/app/board/o9J_l1KiRCc=/)
- teamcity/neo_*.cypher has the script we run to prepare the database, if your test is missing user or get wrong caption, try some script there

### Logging

The logging library [loglevel](https://github.com/pimterry/loglevel) is being applied under the hood, refer to the [docs](https://github.com/pimterry/loglevel#documentation) to see for further information or details if required.

To see all **logging options**, such as setting log levels, printing the logs or downloading log files, open the web browser's console and type `bloom_help()` and hit enter. A list of options is presented.

> All available **levels** and **loggers** are printed when entering `bloom_help()` in the web browser's console.
>
> Available loggers: ROOT, NVL, DRIVER, SSO, PERF
>
> Available levels: trace, debug, info, warn, error, silent

There are also URL query parameters available to set the log levels of individual or all loggers.

- The parameter `LOGS_LEVEL=<level>` will set the logging level to `level` for _all_ loggers
- The parameter `<LOGGER>_LEVEL=<level>` will set the logging level to `level` for the logger `<LOGGER>`

Examples:

`http://localhost:8085?ROOT_LEVEL=warn` set the logging level `warn` for the logger `ROOT` (the application logs)

`http://localhost:8085?LOGS_LEVEL=debug` set the logging level `debug` for the _all_ loggers

`http://localhost:8085?LOGS_LEVEL=error&DRIVER_LEVEL=debug&NVL_LEVEL=info` set the logging level `error` for _all_ loggers yet for the `DRIVER` logger the level is set to `debug` and for the `NVL` logger the level is set to `info`

You are able to _download_ all log files via the Experimental drawer or via a command outlined in `bloom_help()`.

### URL query parameters

Meant for users:
| URL query parameter | Example | Description
| ------ | ------ | ------ |
| discoveryURL=`url` | discoveryURL=https://localhost:8083/discovery.json | See [Bloom docs](https://neo4j.com/docs/bloom-user-guide/current/) |
| connectURL=`url` | connectURL=bolt://localhost:7687 | See [Bloom docs](https://neo4j.com/docs/bloom-user-guide/current/) |
| search=`term` | search=Tom Hanks | See [Bloom docs](https://neo4j.com/docs/bloom-user-guide/current/) |
| perspective=`perspectiveName` | perspective=perspective 12 | See [Bloom docs](https://neo4j.com/docs/bloom-user-guide/current/) |
| run=`boolean` | run=true | See [Bloom docs](https://neo4j.com/docs/bloom-user-guide/current/) |
| sso_redirect=`idp_id` | sso_redirect=keycloak-oidc | Use for the auto-redirect to a SSO provider login page |
| LOGS\_LEVEL=`level` | LOGS_LEVEL=warn | Set the logging level to `level` for all loggers |
| `LOGGER`\_LEVEL=`level` | DRIVER_LEVEL=debug | Set the logging level to `level` for the logger `LOGGER` |

Meant only for developers and application development:
| URL query parameter | Example | Description
| ------ | ------ | ------ |
| logout_timeout=`seconds` | logout_timeout=200 | Sets the logout timeout to `seconds`, meant only for E2E tests |
| grid_layout=`boolean` | grid_layout=true | Sets the visualization layout type to a grid layout, meant only for E2E tests, Default: false |
| ntid=`uuid` | ntid=jdsf-342-sdf-dfsdf | Tracking ID provided by Aura when launching Bloom from within the Aura console |
| auth_flow_step=`arg` | auth_flow_step=redirect_uri | SSO: If the user arrives back to the client application with the URL param auth_flow_step=redirect_uri we know it's time to proceed in the SSO auth process |
| idp_id=`idp_id` | idp_id=keycloak-oidc | SSO: The user should arrive with a URL param named idp_id that we can map to the information in the discovery data to figure out how to proceed |

## Devtools
Download these two chrome extensions:
- [React devtools](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi?hl=en)
