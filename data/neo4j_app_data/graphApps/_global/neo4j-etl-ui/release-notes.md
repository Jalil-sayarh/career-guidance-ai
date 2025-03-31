## 1.6.0

* release to npm instead of jfrog
* fix: When switching to another desktop project the rdbms connection fails
* fix: state change of database is not correctly detected

## 1.5.1

* fix path bug that led to missing `org.neo4j.etl.rdbms.Support`
* re-enable passing long command line arguments via @file
* fix camelCase mapping of property names
* fixed label mapping with fromSQL import
* fixed bug about changing property in fromSQL import
* new [neo4j-etl tool docs](https://neo4j.com/labs/etl-tool/)

## 1.5.0

New Release with Neo4j 4.x compability

* support multi database for offline and online imports
* support new Neo4j Driver 4.0.1
* support direct transfer from relational database without intermediate CSVs
* Desktop 1.2.5 support with new signing key
* Some UI improvements

## 1.4.1

* Allow for other remote connection schemes than bolt:// (i.e. bolt+routing:// and neo4j://)

## 1.4.0

* Add support for remote databases (use tmp directory for csv and mapping)
* Add new import mode required for remote databases using UNWIND and batches of rows
* Fix some SQL type handling

## 1.3.7

* Fix permission issue
* Add ability to skip nodes, relationships and properties
* Fix import mode dropdown initial selection

## 1.3.6

* Fix Image Packaging for Security Info
* Please enable ["Background Process" under the "Security Shield"](https://github.com/neo-technology/neo4j-etl/raw/master/neo4j-etl-ui/src/browser/modules/Stream/JavaPerm.jpg) for the Neo4j-ETL Graph App


## 1.3.5

* Allow to skip fields and remove Nodes / Relationships from Mapping
* Show Information on how to enable background process access in Graph App Settings

## 1.3.3

* Work around Desktop API Issue

## 1.3.2

* Fix issue with custom JDBC driver jar
* Show node labels for relationships

## 1.3.1

* Reimplementation of UI in React
* New, cleaner UI 
* Colors for nodes, edit properties on double-click
* Fix: rel-properties
* Fix: Azure SQL
* Fix: MySQL Auth and new driver bundled
* Fix: Generate CSV with proper quotes

## 1.2.1

* Fix: Datatype conversion
* Fix: auth if disabled
* Fix: quoting for neo4j-import
* Fix: ignore schema casing, default oracle schema
* Fix: CLI skip tables by pattern

