/******/ (function(modules) { // webpackBootstrap
/******/ 	// install a JSONP callback for chunk loading
/******/ 	function webpackJsonpCallback(data) {
/******/ 		var chunkIds = data[0];
/******/ 		var moreModules = data[1];
/******/ 		var executeModules = data[2];
/******/
/******/ 		// add "moreModules" to the modules object,
/******/ 		// then flag all "chunkIds" as loaded and fire callback
/******/ 		var moduleId, chunkId, i = 0, resolves = [];
/******/ 		for(;i < chunkIds.length; i++) {
/******/ 			chunkId = chunkIds[i];
/******/ 			if(Object.prototype.hasOwnProperty.call(installedChunks, chunkId) && installedChunks[chunkId]) {
/******/ 				resolves.push(installedChunks[chunkId][0]);
/******/ 			}
/******/ 			installedChunks[chunkId] = 0;
/******/ 		}
/******/ 		for(moduleId in moreModules) {
/******/ 			if(Object.prototype.hasOwnProperty.call(moreModules, moduleId)) {
/******/ 				modules[moduleId] = moreModules[moduleId];
/******/ 			}
/******/ 		}
/******/ 		if(parentJsonpFunction) parentJsonpFunction(data);
/******/
/******/ 		while(resolves.length) {
/******/ 			resolves.shift()();
/******/ 		}
/******/
/******/ 		// add entry modules from loaded chunk to deferred list
/******/ 		deferredModules.push.apply(deferredModules, executeModules || []);
/******/
/******/ 		// run deferred modules when all chunks ready
/******/ 		return checkDeferredModules();
/******/ 	};
/******/ 	function checkDeferredModules() {
/******/ 		var result;
/******/ 		for(var i = 0; i < deferredModules.length; i++) {
/******/ 			var deferredModule = deferredModules[i];
/******/ 			var fulfilled = true;
/******/ 			for(var j = 1; j < deferredModule.length; j++) {
/******/ 				var depId = deferredModule[j];
/******/ 				if(installedChunks[depId] !== 0) fulfilled = false;
/******/ 			}
/******/ 			if(fulfilled) {
/******/ 				deferredModules.splice(i--, 1);
/******/ 				result = __webpack_require__(__webpack_require__.s = deferredModule[0]);
/******/ 			}
/******/ 		}
/******/
/******/ 		return result;
/******/ 	}
/******/
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// object to store loaded and loading chunks
/******/ 	// undefined = chunk not loaded, null = chunk preloaded/prefetched
/******/ 	// Promise = chunk loading, 0 = chunk loaded
/******/ 	var installedChunks = {
/******/ 		0: 0
/******/ 	};
/******/
/******/ 	var deferredModules = [];
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	var jsonpArray = window["webpackJsonp"] = window["webpackJsonp"] || [];
/******/ 	var oldJsonpFunction = jsonpArray.push.bind(jsonpArray);
/******/ 	jsonpArray.push = webpackJsonpCallback;
/******/ 	jsonpArray = jsonpArray.slice();
/******/ 	for(var i = 0; i < jsonpArray.length; i++) webpackJsonpCallback(jsonpArray[i]);
/******/ 	var parentJsonpFunction = oldJsonpFunction;
/******/
/******/
/******/ 	// add entry module to deferred list
/******/ 	deferredModules.push([0,2,1]);
/******/ 	// run deferred modules when ready
/******/ 	return checkDeferredModules();
/******/ })
/************************************************************************/
/******/ ({

/***/ "../../node_modules/css-loader/index.js!./styles/font-awesome.min.css":
/***/ (function(module, exports, __webpack_require__) {

var escape = __webpack_require__("../../node_modules/css-loader/lib/url/escape.js");
exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, "/*!\n *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome\n *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)\n */@font-face{font-family:FontAwesome;src:url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.eot?v=4.7.0")) + ");src:url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.eot")) + "?#iefix&v=4.7.0) format(\"embedded-opentype\"),url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.woff2?v=4.7.0")) + ") format(\"woff2\"),url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.woff?v=4.7.0")) + ") format(\"woff\"),url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.ttf?v=4.7.0")) + ") format(\"truetype\"),url(" + escape(__webpack_require__("./styles/fonts/fontawesome-webfont.svg?v=4.7.0")) + "#fontawesomeregular) format(\"svg\");font-weight:400;font-style:normal}.fa{display:inline-block;font:normal normal normal 14px/1 FontAwesome;font-size:inherit;text-rendering:auto;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}.fa-lg{font-size:1.33333333em;line-height:.75em;vertical-align:-15%}.fa-2x{font-size:2em}.fa-3x{font-size:3em}.fa-4x{font-size:4em}.fa-5x{font-size:5em}.fa-fw{width:1.28571429em;text-align:center}.fa-ul{padding-left:0;margin-left:2.14285714em;list-style-type:none}.fa-ul>li{position:relative}.fa-li{position:absolute;left:-2.14285714em;width:2.14285714em;top:.14285714em;text-align:center}.fa-li.fa-lg{left:-1.85714286em}.fa-border{padding:.2em .25em .15em;border:.08em solid #eee;border-radius:.1em}.fa-pull-left{float:left}.fa-pull-right{float:right}.fa.fa-pull-left{margin-right:.3em}.fa.fa-pull-right{margin-left:.3em}.pull-right{float:right}.pull-left{float:left}.fa.pull-left{margin-right:.3em}.fa.pull-right{margin-left:.3em}.fa-spin{-webkit-animation:fa-spin 2s infinite linear;animation:fa-spin 2s infinite linear}.fa-pulse{-webkit-animation:fa-spin 1s infinite steps(8);animation:fa-spin 1s infinite steps(8)}@-webkit-keyframes fa-spin{0%{-webkit-transform:rotate(0deg);transform:rotate(0deg)}to{-webkit-transform:rotate(359deg);transform:rotate(359deg)}}@keyframes fa-spin{0%{-webkit-transform:rotate(0deg);transform:rotate(0deg)}to{-webkit-transform:rotate(359deg);transform:rotate(359deg)}}.fa-rotate-90{-ms-filter:\"progid:DXImageTransform.Microsoft.BasicImage(rotation=1)\";-webkit-transform:rotate(90deg);-ms-transform:rotate(90deg);transform:rotate(90deg)}.fa-rotate-180{-ms-filter:\"progid:DXImageTransform.Microsoft.BasicImage(rotation=2)\";-webkit-transform:rotate(180deg);-ms-transform:rotate(180deg);transform:rotate(180deg)}.fa-rotate-270{-ms-filter:\"progid:DXImageTransform.Microsoft.BasicImage(rotation=3)\";-webkit-transform:rotate(270deg);-ms-transform:rotate(270deg);transform:rotate(270deg)}.fa-flip-horizontal{-ms-filter:\"progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)\";-webkit-transform:scaleX(-1);-ms-transform:scaleX(-1);transform:scaleX(-1)}.fa-flip-vertical{-ms-filter:\"progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)\";-webkit-transform:scaleY(-1);-ms-transform:scaleY(-1);transform:scaleY(-1)}:root .fa-flip-horizontal,:root .fa-flip-vertical,:root .fa-rotate-90,:root .fa-rotate-180,:root .fa-rotate-270{filter:none}.fa-stack{position:relative;display:inline-block;width:2em;height:2em;line-height:2em;vertical-align:middle}.fa-stack-1x,.fa-stack-2x{position:absolute;left:0;width:100%;text-align:center}.fa-stack-1x{line-height:inherit}.fa-stack-2x{font-size:2em}.fa-inverse{color:#fff}.fa-glass:before{content:\"\\F000\"}.fa-music:before{content:\"\\F001\"}.fa-search:before{content:\"\\F002\"}.fa-envelope-o:before{content:\"\\F003\"}.fa-heart:before{content:\"\\F004\"}.fa-star:before{content:\"\\F005\"}.fa-star-o:before{content:\"\\F006\"}.fa-user:before{content:\"\\F007\"}.fa-film:before{content:\"\\F008\"}.fa-th-large:before{content:\"\\F009\"}.fa-th:before{content:\"\\F00A\"}.fa-th-list:before{content:\"\\F00B\"}.fa-check:before{content:\"\\F00C\"}.fa-close:before,.fa-remove:before,.fa-times:before{content:\"\\F00D\"}.fa-search-plus:before{content:\"\\F00E\"}.fa-search-minus:before{content:\"\\F010\"}.fa-power-off:before{content:\"\\F011\"}.fa-signal:before{content:\"\\F012\"}.fa-cog:before,.fa-gear:before{content:\"\\F013\"}.fa-trash-o:before{content:\"\\F014\"}.fa-home:before{content:\"\\F015\"}.fa-file-o:before{content:\"\\F016\"}.fa-clock-o:before{content:\"\\F017\"}.fa-road:before{content:\"\\F018\"}.fa-download:before{content:\"\\F019\"}.fa-arrow-circle-o-down:before{content:\"\\F01A\"}.fa-arrow-circle-o-up:before{content:\"\\F01B\"}.fa-inbox:before{content:\"\\F01C\"}.fa-play-circle-o:before{content:\"\\F01D\"}.fa-repeat:before,.fa-rotate-right:before{content:\"\\F01E\"}.fa-refresh:before{content:\"\\F021\"}.fa-list-alt:before{content:\"\\F022\"}.fa-lock:before{content:\"\\F023\"}.fa-flag:before{content:\"\\F024\"}.fa-headphones:before{content:\"\\F025\"}.fa-volume-off:before{content:\"\\F026\"}.fa-volume-down:before{content:\"\\F027\"}.fa-volume-up:before{content:\"\\F028\"}.fa-qrcode:before{content:\"\\F029\"}.fa-barcode:before{content:\"\\F02A\"}.fa-tag:before{content:\"\\F02B\"}.fa-tags:before{content:\"\\F02C\"}.fa-book:before{content:\"\\F02D\"}.fa-bookmark:before{content:\"\\F02E\"}.fa-print:before{content:\"\\F02F\"}.fa-camera:before{content:\"\\F030\"}.fa-font:before{content:\"\\F031\"}.fa-bold:before{content:\"\\F032\"}.fa-italic:before{content:\"\\F033\"}.fa-text-height:before{content:\"\\F034\"}.fa-text-width:before{content:\"\\F035\"}.fa-align-left:before{content:\"\\F036\"}.fa-align-center:before{content:\"\\F037\"}.fa-align-right:before{content:\"\\F038\"}.fa-align-justify:before{content:\"\\F039\"}.fa-list:before{content:\"\\F03A\"}.fa-dedent:before,.fa-outdent:before{content:\"\\F03B\"}.fa-indent:before{content:\"\\F03C\"}.fa-video-camera:before{content:\"\\F03D\"}.fa-image:before,.fa-photo:before,.fa-picture-o:before{content:\"\\F03E\"}.fa-pencil:before{content:\"\\F040\"}.fa-map-marker:before{content:\"\\F041\"}.fa-adjust:before{content:\"\\F042\"}.fa-tint:before{content:\"\\F043\"}.fa-edit:before,.fa-pencil-square-o:before{content:\"\\F044\"}.fa-share-square-o:before{content:\"\\F045\"}.fa-check-square-o:before{content:\"\\F046\"}.fa-arrows:before{content:\"\\F047\"}.fa-step-backward:before{content:\"\\F048\"}.fa-fast-backward:before{content:\"\\F049\"}.fa-backward:before{content:\"\\F04A\"}.fa-play:before{content:\"\\F04B\"}.fa-pause:before{content:\"\\F04C\"}.fa-stop:before{content:\"\\F04D\"}.fa-forward:before{content:\"\\F04E\"}.fa-fast-forward:before{content:\"\\F050\"}.fa-step-forward:before{content:\"\\F051\"}.fa-eject:before{content:\"\\F052\"}.fa-chevron-left:before{content:\"\\F053\"}.fa-chevron-right:before{content:\"\\F054\"}.fa-plus-circle:before{content:\"\\F055\"}.fa-minus-circle:before{content:\"\\F056\"}.fa-times-circle:before{content:\"\\F057\"}.fa-check-circle:before{content:\"\\F058\"}.fa-question-circle:before{content:\"\\F059\"}.fa-info-circle:before{content:\"\\F05A\"}.fa-crosshairs:before{content:\"\\F05B\"}.fa-times-circle-o:before{content:\"\\F05C\"}.fa-check-circle-o:before{content:\"\\F05D\"}.fa-ban:before{content:\"\\F05E\"}.fa-arrow-left:before{content:\"\\F060\"}.fa-arrow-right:before{content:\"\\F061\"}.fa-arrow-up:before{content:\"\\F062\"}.fa-arrow-down:before{content:\"\\F063\"}.fa-mail-forward:before,.fa-share:before{content:\"\\F064\"}.fa-expand:before{content:\"\\F065\"}.fa-compress:before{content:\"\\F066\"}.fa-plus:before{content:\"\\F067\"}.fa-minus:before{content:\"\\F068\"}.fa-asterisk:before{content:\"\\F069\"}.fa-exclamation-circle:before{content:\"\\F06A\"}.fa-gift:before{content:\"\\F06B\"}.fa-leaf:before{content:\"\\F06C\"}.fa-fire:before{content:\"\\F06D\"}.fa-eye:before{content:\"\\F06E\"}.fa-eye-slash:before{content:\"\\F070\"}.fa-exclamation-triangle:before,.fa-warning:before{content:\"\\F071\"}.fa-plane:before{content:\"\\F072\"}.fa-calendar:before{content:\"\\F073\"}.fa-random:before{content:\"\\F074\"}.fa-comment:before{content:\"\\F075\"}.fa-magnet:before{content:\"\\F076\"}.fa-chevron-up:before{content:\"\\F077\"}.fa-chevron-down:before{content:\"\\F078\"}.fa-retweet:before{content:\"\\F079\"}.fa-shopping-cart:before{content:\"\\F07A\"}.fa-folder:before{content:\"\\F07B\"}.fa-folder-open:before{content:\"\\F07C\"}.fa-arrows-v:before{content:\"\\F07D\"}.fa-arrows-h:before{content:\"\\F07E\"}.fa-bar-chart-o:before,.fa-bar-chart:before{content:\"\\F080\"}.fa-twitter-square:before{content:\"\\F081\"}.fa-facebook-square:before{content:\"\\F082\"}.fa-camera-retro:before{content:\"\\F083\"}.fa-key:before{content:\"\\F084\"}.fa-cogs:before,.fa-gears:before{content:\"\\F085\"}.fa-comments:before{content:\"\\F086\"}.fa-thumbs-o-up:before{content:\"\\F087\"}.fa-thumbs-o-down:before{content:\"\\F088\"}.fa-star-half:before{content:\"\\F089\"}.fa-heart-o:before{content:\"\\F08A\"}.fa-sign-out:before{content:\"\\F08B\"}.fa-linkedin-square:before{content:\"\\F08C\"}.fa-thumb-tack:before{content:\"\\F08D\"}.fa-external-link:before{content:\"\\F08E\"}.fa-sign-in:before{content:\"\\F090\"}.fa-trophy:before{content:\"\\F091\"}.fa-github-square:before{content:\"\\F092\"}.fa-upload:before{content:\"\\F093\"}.fa-lemon-o:before{content:\"\\F094\"}.fa-phone:before{content:\"\\F095\"}.fa-square-o:before{content:\"\\F096\"}.fa-bookmark-o:before{content:\"\\F097\"}.fa-phone-square:before{content:\"\\F098\"}.fa-twitter:before{content:\"\\F099\"}.fa-facebook-f:before,.fa-facebook:before{content:\"\\F09A\"}.fa-github:before{content:\"\\F09B\"}.fa-unlock:before{content:\"\\F09C\"}.fa-credit-card:before{content:\"\\F09D\"}.fa-feed:before,.fa-rss:before{content:\"\\F09E\"}.fa-hdd-o:before{content:\"\\F0A0\"}.fa-bullhorn:before{content:\"\\F0A1\"}.fa-bell:before{content:\"\\F0F3\"}.fa-certificate:before{content:\"\\F0A3\"}.fa-hand-o-right:before{content:\"\\F0A4\"}.fa-hand-o-left:before{content:\"\\F0A5\"}.fa-hand-o-up:before{content:\"\\F0A6\"}.fa-hand-o-down:before{content:\"\\F0A7\"}.fa-arrow-circle-left:before{content:\"\\F0A8\"}.fa-arrow-circle-right:before{content:\"\\F0A9\"}.fa-arrow-circle-up:before{content:\"\\F0AA\"}.fa-arrow-circle-down:before{content:\"\\F0AB\"}.fa-globe:before{content:\"\\F0AC\"}.fa-wrench:before{content:\"\\F0AD\"}.fa-tasks:before{content:\"\\F0AE\"}.fa-filter:before{content:\"\\F0B0\"}.fa-briefcase:before{content:\"\\F0B1\"}.fa-arrows-alt:before{content:\"\\F0B2\"}.fa-group:before,.fa-users:before{content:\"\\F0C0\"}.fa-chain:before,.fa-link:before{content:\"\\F0C1\"}.fa-cloud:before{content:\"\\F0C2\"}.fa-flask:before{content:\"\\F0C3\"}.fa-cut:before,.fa-scissors:before{content:\"\\F0C4\"}.fa-copy:before,.fa-files-o:before{content:\"\\F0C5\"}.fa-paperclip:before{content:\"\\F0C6\"}.fa-floppy-o:before,.fa-save:before{content:\"\\F0C7\"}.fa-square:before{content:\"\\F0C8\"}.fa-bars:before,.fa-navicon:before,.fa-reorder:before{content:\"\\F0C9\"}.fa-list-ul:before{content:\"\\F0CA\"}.fa-list-ol:before{content:\"\\F0CB\"}.fa-strikethrough:before{content:\"\\F0CC\"}.fa-underline:before{content:\"\\F0CD\"}.fa-table:before{content:\"\\F0CE\"}.fa-magic:before{content:\"\\F0D0\"}.fa-truck:before{content:\"\\F0D1\"}.fa-pinterest:before{content:\"\\F0D2\"}.fa-pinterest-square:before{content:\"\\F0D3\"}.fa-google-plus-square:before{content:\"\\F0D4\"}.fa-google-plus:before{content:\"\\F0D5\"}.fa-money:before{content:\"\\F0D6\"}.fa-caret-down:before{content:\"\\F0D7\"}.fa-caret-up:before{content:\"\\F0D8\"}.fa-caret-left:before{content:\"\\F0D9\"}.fa-caret-right:before{content:\"\\F0DA\"}.fa-columns:before{content:\"\\F0DB\"}.fa-sort:before,.fa-unsorted:before{content:\"\\F0DC\"}.fa-sort-desc:before,.fa-sort-down:before{content:\"\\F0DD\"}.fa-sort-asc:before,.fa-sort-up:before{content:\"\\F0DE\"}.fa-envelope:before{content:\"\\F0E0\"}.fa-linkedin:before{content:\"\\F0E1\"}.fa-rotate-left:before,.fa-undo:before{content:\"\\F0E2\"}.fa-gavel:before,.fa-legal:before{content:\"\\F0E3\"}.fa-dashboard:before,.fa-tachometer:before{content:\"\\F0E4\"}.fa-comment-o:before{content:\"\\F0E5\"}.fa-comments-o:before{content:\"\\F0E6\"}.fa-bolt:before,.fa-flash:before{content:\"\\F0E7\"}.fa-sitemap:before{content:\"\\F0E8\"}.fa-umbrella:before{content:\"\\F0E9\"}.fa-clipboard:before,.fa-paste:before{content:\"\\F0EA\"}.fa-lightbulb-o:before{content:\"\\F0EB\"}.fa-exchange:before{content:\"\\F0EC\"}.fa-cloud-download:before{content:\"\\F0ED\"}.fa-cloud-upload:before{content:\"\\F0EE\"}.fa-user-md:before{content:\"\\F0F0\"}.fa-stethoscope:before{content:\"\\F0F1\"}.fa-suitcase:before{content:\"\\F0F2\"}.fa-bell-o:before{content:\"\\F0A2\"}.fa-coffee:before{content:\"\\F0F4\"}.fa-cutlery:before{content:\"\\F0F5\"}.fa-file-text-o:before{content:\"\\F0F6\"}.fa-building-o:before{content:\"\\F0F7\"}.fa-hospital-o:before{content:\"\\F0F8\"}.fa-ambulance:before{content:\"\\F0F9\"}.fa-medkit:before{content:\"\\F0FA\"}.fa-fighter-jet:before{content:\"\\F0FB\"}.fa-beer:before{content:\"\\F0FC\"}.fa-h-square:before{content:\"\\F0FD\"}.fa-plus-square:before{content:\"\\F0FE\"}.fa-angle-double-left:before{content:\"\\F100\"}.fa-angle-double-right:before{content:\"\\F101\"}.fa-angle-double-up:before{content:\"\\F102\"}.fa-angle-double-down:before{content:\"\\F103\"}.fa-angle-left:before{content:\"\\F104\"}.fa-angle-right:before{content:\"\\F105\"}.fa-angle-up:before{content:\"\\F106\"}.fa-angle-down:before{content:\"\\F107\"}.fa-desktop:before{content:\"\\F108\"}.fa-laptop:before{content:\"\\F109\"}.fa-tablet:before{content:\"\\F10A\"}.fa-mobile-phone:before,.fa-mobile:before{content:\"\\F10B\"}.fa-circle-o:before{content:\"\\F10C\"}.fa-quote-left:before{content:\"\\F10D\"}.fa-quote-right:before{content:\"\\F10E\"}.fa-spinner:before{content:\"\\F110\"}.fa-circle:before{content:\"\\F111\"}.fa-mail-reply:before,.fa-reply:before{content:\"\\F112\"}.fa-github-alt:before{content:\"\\F113\"}.fa-folder-o:before{content:\"\\F114\"}.fa-folder-open-o:before{content:\"\\F115\"}.fa-smile-o:before{content:\"\\F118\"}.fa-frown-o:before{content:\"\\F119\"}.fa-meh-o:before{content:\"\\F11A\"}.fa-gamepad:before{content:\"\\F11B\"}.fa-keyboard-o:before{content:\"\\F11C\"}.fa-flag-o:before{content:\"\\F11D\"}.fa-flag-checkered:before{content:\"\\F11E\"}.fa-terminal:before{content:\"\\F120\"}.fa-code:before{content:\"\\F121\"}.fa-mail-reply-all:before,.fa-reply-all:before{content:\"\\F122\"}.fa-star-half-empty:before,.fa-star-half-full:before,.fa-star-half-o:before{content:\"\\F123\"}.fa-location-arrow:before{content:\"\\F124\"}.fa-crop:before{content:\"\\F125\"}.fa-code-fork:before{content:\"\\F126\"}.fa-chain-broken:before,.fa-unlink:before{content:\"\\F127\"}.fa-question:before{content:\"\\F128\"}.fa-info:before{content:\"\\F129\"}.fa-exclamation:before{content:\"\\F12A\"}.fa-superscript:before{content:\"\\F12B\"}.fa-subscript:before{content:\"\\F12C\"}.fa-eraser:before{content:\"\\F12D\"}.fa-puzzle-piece:before{content:\"\\F12E\"}.fa-microphone:before{content:\"\\F130\"}.fa-microphone-slash:before{content:\"\\F131\"}.fa-shield:before{content:\"\\F132\"}.fa-calendar-o:before{content:\"\\F133\"}.fa-fire-extinguisher:before{content:\"\\F134\"}.fa-rocket:before{content:\"\\F135\"}.fa-maxcdn:before{content:\"\\F136\"}.fa-chevron-circle-left:before{content:\"\\F137\"}.fa-chevron-circle-right:before{content:\"\\F138\"}.fa-chevron-circle-up:before{content:\"\\F139\"}.fa-chevron-circle-down:before{content:\"\\F13A\"}.fa-html5:before{content:\"\\F13B\"}.fa-css3:before{content:\"\\F13C\"}.fa-anchor:before{content:\"\\F13D\"}.fa-unlock-alt:before{content:\"\\F13E\"}.fa-bullseye:before{content:\"\\F140\"}.fa-ellipsis-h:before{content:\"\\F141\"}.fa-ellipsis-v:before{content:\"\\F142\"}.fa-rss-square:before{content:\"\\F143\"}.fa-play-circle:before{content:\"\\F144\"}.fa-ticket:before{content:\"\\F145\"}.fa-minus-square:before{content:\"\\F146\"}.fa-minus-square-o:before{content:\"\\F147\"}.fa-level-up:before{content:\"\\F148\"}.fa-level-down:before{content:\"\\F149\"}.fa-check-square:before{content:\"\\F14A\"}.fa-pencil-square:before{content:\"\\F14B\"}.fa-external-link-square:before{content:\"\\F14C\"}.fa-share-square:before{content:\"\\F14D\"}.fa-compass:before{content:\"\\F14E\"}.fa-caret-square-o-down:before,.fa-toggle-down:before{content:\"\\F150\"}.fa-caret-square-o-up:before,.fa-toggle-up:before{content:\"\\F151\"}.fa-caret-square-o-right:before,.fa-toggle-right:before{content:\"\\F152\"}.fa-eur:before,.fa-euro:before{content:\"\\F153\"}.fa-gbp:before{content:\"\\F154\"}.fa-dollar:before,.fa-usd:before{content:\"\\F155\"}.fa-inr:before,.fa-rupee:before{content:\"\\F156\"}.fa-cny:before,.fa-jpy:before,.fa-rmb:before,.fa-yen:before{content:\"\\F157\"}.fa-rouble:before,.fa-rub:before,.fa-ruble:before{content:\"\\F158\"}.fa-krw:before,.fa-won:before{content:\"\\F159\"}.fa-bitcoin:before,.fa-btc:before{content:\"\\F15A\"}.fa-file:before{content:\"\\F15B\"}.fa-file-text:before{content:\"\\F15C\"}.fa-sort-alpha-asc:before{content:\"\\F15D\"}.fa-sort-alpha-desc:before{content:\"\\F15E\"}.fa-sort-amount-asc:before{content:\"\\F160\"}.fa-sort-amount-desc:before{content:\"\\F161\"}.fa-sort-numeric-asc:before{content:\"\\F162\"}.fa-sort-numeric-desc:before{content:\"\\F163\"}.fa-thumbs-up:before{content:\"\\F164\"}.fa-thumbs-down:before{content:\"\\F165\"}.fa-youtube-square:before{content:\"\\F166\"}.fa-youtube:before{content:\"\\F167\"}.fa-xing:before{content:\"\\F168\"}.fa-xing-square:before{content:\"\\F169\"}.fa-youtube-play:before{content:\"\\F16A\"}.fa-dropbox:before{content:\"\\F16B\"}.fa-stack-overflow:before{content:\"\\F16C\"}.fa-instagram:before{content:\"\\F16D\"}.fa-flickr:before{content:\"\\F16E\"}.fa-adn:before{content:\"\\F170\"}.fa-bitbucket:before{content:\"\\F171\"}.fa-bitbucket-square:before{content:\"\\F172\"}.fa-tumblr:before{content:\"\\F173\"}.fa-tumblr-square:before{content:\"\\F174\"}.fa-long-arrow-down:before{content:\"\\F175\"}.fa-long-arrow-up:before{content:\"\\F176\"}.fa-long-arrow-left:before{content:\"\\F177\"}.fa-long-arrow-right:before{content:\"\\F178\"}.fa-apple:before{content:\"\\F179\"}.fa-windows:before{content:\"\\F17A\"}.fa-android:before{content:\"\\F17B\"}.fa-linux:before{content:\"\\F17C\"}.fa-dribbble:before{content:\"\\F17D\"}.fa-skype:before{content:\"\\F17E\"}.fa-foursquare:before{content:\"\\F180\"}.fa-trello:before{content:\"\\F181\"}.fa-female:before{content:\"\\F182\"}.fa-male:before{content:\"\\F183\"}.fa-gittip:before,.fa-gratipay:before{content:\"\\F184\"}.fa-sun-o:before{content:\"\\F185\"}.fa-moon-o:before{content:\"\\F186\"}.fa-archive:before{content:\"\\F187\"}.fa-bug:before{content:\"\\F188\"}.fa-vk:before{content:\"\\F189\"}.fa-weibo:before{content:\"\\F18A\"}.fa-renren:before{content:\"\\F18B\"}.fa-pagelines:before{content:\"\\F18C\"}.fa-stack-exchange:before{content:\"\\F18D\"}.fa-arrow-circle-o-right:before{content:\"\\F18E\"}.fa-arrow-circle-o-left:before{content:\"\\F190\"}.fa-caret-square-o-left:before,.fa-toggle-left:before{content:\"\\F191\"}.fa-dot-circle-o:before{content:\"\\F192\"}.fa-wheelchair:before{content:\"\\F193\"}.fa-vimeo-square:before{content:\"\\F194\"}.fa-try:before,.fa-turkish-lira:before{content:\"\\F195\"}.fa-plus-square-o:before{content:\"\\F196\"}.fa-space-shuttle:before{content:\"\\F197\"}.fa-slack:before{content:\"\\F198\"}.fa-envelope-square:before{content:\"\\F199\"}.fa-wordpress:before{content:\"\\F19A\"}.fa-openid:before{content:\"\\F19B\"}.fa-bank:before,.fa-institution:before,.fa-university:before{content:\"\\F19C\"}.fa-graduation-cap:before,.fa-mortar-board:before{content:\"\\F19D\"}.fa-yahoo:before{content:\"\\F19E\"}.fa-google:before{content:\"\\F1A0\"}.fa-reddit:before{content:\"\\F1A1\"}.fa-reddit-square:before{content:\"\\F1A2\"}.fa-stumbleupon-circle:before{content:\"\\F1A3\"}.fa-stumbleupon:before{content:\"\\F1A4\"}.fa-delicious:before{content:\"\\F1A5\"}.fa-digg:before{content:\"\\F1A6\"}.fa-pied-piper-pp:before{content:\"\\F1A7\"}.fa-pied-piper-alt:before{content:\"\\F1A8\"}.fa-drupal:before{content:\"\\F1A9\"}.fa-joomla:before{content:\"\\F1AA\"}.fa-language:before{content:\"\\F1AB\"}.fa-fax:before{content:\"\\F1AC\"}.fa-building:before{content:\"\\F1AD\"}.fa-child:before{content:\"\\F1AE\"}.fa-paw:before{content:\"\\F1B0\"}.fa-spoon:before{content:\"\\F1B1\"}.fa-cube:before{content:\"\\F1B2\"}.fa-cubes:before{content:\"\\F1B3\"}.fa-behance:before{content:\"\\F1B4\"}.fa-behance-square:before{content:\"\\F1B5\"}.fa-steam:before{content:\"\\F1B6\"}.fa-steam-square:before{content:\"\\F1B7\"}.fa-recycle:before{content:\"\\F1B8\"}.fa-automobile:before,.fa-car:before{content:\"\\F1B9\"}.fa-cab:before,.fa-taxi:before{content:\"\\F1BA\"}.fa-tree:before{content:\"\\F1BB\"}.fa-spotify:before{content:\"\\F1BC\"}.fa-deviantart:before{content:\"\\F1BD\"}.fa-soundcloud:before{content:\"\\F1BE\"}.fa-database:before{content:\"\\F1C0\"}.fa-file-pdf-o:before{content:\"\\F1C1\"}.fa-file-word-o:before{content:\"\\F1C2\"}.fa-file-excel-o:before{content:\"\\F1C3\"}.fa-file-powerpoint-o:before{content:\"\\F1C4\"}.fa-file-image-o:before,.fa-file-photo-o:before,.fa-file-picture-o:before{content:\"\\F1C5\"}.fa-file-archive-o:before,.fa-file-zip-o:before{content:\"\\F1C6\"}.fa-file-audio-o:before,.fa-file-sound-o:before{content:\"\\F1C7\"}.fa-file-movie-o:before,.fa-file-video-o:before{content:\"\\F1C8\"}.fa-file-code-o:before{content:\"\\F1C9\"}.fa-vine:before{content:\"\\F1CA\"}.fa-codepen:before{content:\"\\F1CB\"}.fa-jsfiddle:before{content:\"\\F1CC\"}.fa-life-bouy:before,.fa-life-buoy:before,.fa-life-ring:before,.fa-life-saver:before,.fa-support:before{content:\"\\F1CD\"}.fa-circle-o-notch:before{content:\"\\F1CE\"}.fa-ra:before,.fa-rebel:before,.fa-resistance:before{content:\"\\F1D0\"}.fa-empire:before,.fa-ge:before{content:\"\\F1D1\"}.fa-git-square:before{content:\"\\F1D2\"}.fa-git:before{content:\"\\F1D3\"}.fa-hacker-news:before,.fa-y-combinator-square:before,.fa-yc-square:before{content:\"\\F1D4\"}.fa-tencent-weibo:before{content:\"\\F1D5\"}.fa-qq:before{content:\"\\F1D6\"}.fa-wechat:before,.fa-weixin:before{content:\"\\F1D7\"}.fa-paper-plane:before,.fa-send:before{content:\"\\F1D8\"}.fa-paper-plane-o:before,.fa-send-o:before{content:\"\\F1D9\"}.fa-history:before{content:\"\\F1DA\"}.fa-circle-thin:before{content:\"\\F1DB\"}.fa-header:before{content:\"\\F1DC\"}.fa-paragraph:before{content:\"\\F1DD\"}.fa-sliders:before{content:\"\\F1DE\"}.fa-share-alt:before{content:\"\\F1E0\"}.fa-share-alt-square:before{content:\"\\F1E1\"}.fa-bomb:before{content:\"\\F1E2\"}.fa-futbol-o:before,.fa-soccer-ball-o:before{content:\"\\F1E3\"}.fa-tty:before{content:\"\\F1E4\"}.fa-binoculars:before{content:\"\\F1E5\"}.fa-plug:before{content:\"\\F1E6\"}.fa-slideshare:before{content:\"\\F1E7\"}.fa-twitch:before{content:\"\\F1E8\"}.fa-yelp:before{content:\"\\F1E9\"}.fa-newspaper-o:before{content:\"\\F1EA\"}.fa-wifi:before{content:\"\\F1EB\"}.fa-calculator:before{content:\"\\F1EC\"}.fa-paypal:before{content:\"\\F1ED\"}.fa-google-wallet:before{content:\"\\F1EE\"}.fa-cc-visa:before{content:\"\\F1F0\"}.fa-cc-mastercard:before{content:\"\\F1F1\"}.fa-cc-discover:before{content:\"\\F1F2\"}.fa-cc-amex:before{content:\"\\F1F3\"}.fa-cc-paypal:before{content:\"\\F1F4\"}.fa-cc-stripe:before{content:\"\\F1F5\"}.fa-bell-slash:before{content:\"\\F1F6\"}.fa-bell-slash-o:before{content:\"\\F1F7\"}.fa-trash:before{content:\"\\F1F8\"}.fa-copyright:before{content:\"\\F1F9\"}.fa-at:before{content:\"\\F1FA\"}.fa-eyedropper:before{content:\"\\F1FB\"}.fa-paint-brush:before{content:\"\\F1FC\"}.fa-birthday-cake:before{content:\"\\F1FD\"}.fa-area-chart:before{content:\"\\F1FE\"}.fa-pie-chart:before{content:\"\\F200\"}.fa-line-chart:before{content:\"\\F201\"}.fa-lastfm:before{content:\"\\F202\"}.fa-lastfm-square:before{content:\"\\F203\"}.fa-toggle-off:before{content:\"\\F204\"}.fa-toggle-on:before{content:\"\\F205\"}.fa-bicycle:before{content:\"\\F206\"}.fa-bus:before{content:\"\\F207\"}.fa-ioxhost:before{content:\"\\F208\"}.fa-angellist:before{content:\"\\F209\"}.fa-cc:before{content:\"\\F20A\"}.fa-ils:before,.fa-shekel:before,.fa-sheqel:before{content:\"\\F20B\"}.fa-meanpath:before{content:\"\\F20C\"}.fa-buysellads:before{content:\"\\F20D\"}.fa-connectdevelop:before{content:\"\\F20E\"}.fa-dashcube:before{content:\"\\F210\"}.fa-forumbee:before{content:\"\\F211\"}.fa-leanpub:before{content:\"\\F212\"}.fa-sellsy:before{content:\"\\F213\"}.fa-shirtsinbulk:before{content:\"\\F214\"}.fa-simplybuilt:before{content:\"\\F215\"}.fa-skyatlas:before{content:\"\\F216\"}.fa-cart-plus:before{content:\"\\F217\"}.fa-cart-arrow-down:before{content:\"\\F218\"}.fa-diamond:before{content:\"\\F219\"}.fa-ship:before{content:\"\\F21A\"}.fa-user-secret:before{content:\"\\F21B\"}.fa-motorcycle:before{content:\"\\F21C\"}.fa-street-view:before{content:\"\\F21D\"}.fa-heartbeat:before{content:\"\\F21E\"}.fa-venus:before{content:\"\\F221\"}.fa-mars:before{content:\"\\F222\"}.fa-mercury:before{content:\"\\F223\"}.fa-intersex:before,.fa-transgender:before{content:\"\\F224\"}.fa-transgender-alt:before{content:\"\\F225\"}.fa-venus-double:before{content:\"\\F226\"}.fa-mars-double:before{content:\"\\F227\"}.fa-venus-mars:before{content:\"\\F228\"}.fa-mars-stroke:before{content:\"\\F229\"}.fa-mars-stroke-v:before{content:\"\\F22A\"}.fa-mars-stroke-h:before{content:\"\\F22B\"}.fa-neuter:before{content:\"\\F22C\"}.fa-genderless:before{content:\"\\F22D\"}.fa-facebook-official:before{content:\"\\F230\"}.fa-pinterest-p:before{content:\"\\F231\"}.fa-whatsapp:before{content:\"\\F232\"}.fa-server:before{content:\"\\F233\"}.fa-user-plus:before{content:\"\\F234\"}.fa-user-times:before{content:\"\\F235\"}.fa-bed:before,.fa-hotel:before{content:\"\\F236\"}.fa-viacoin:before{content:\"\\F237\"}.fa-train:before{content:\"\\F238\"}.fa-subway:before{content:\"\\F239\"}.fa-medium:before{content:\"\\F23A\"}.fa-y-combinator:before,.fa-yc:before{content:\"\\F23B\"}.fa-optin-monster:before{content:\"\\F23C\"}.fa-opencart:before{content:\"\\F23D\"}.fa-expeditedssl:before{content:\"\\F23E\"}.fa-battery-4:before,.fa-battery-full:before,.fa-battery:before{content:\"\\F240\"}.fa-battery-3:before,.fa-battery-three-quarters:before{content:\"\\F241\"}.fa-battery-2:before,.fa-battery-half:before{content:\"\\F242\"}.fa-battery-1:before,.fa-battery-quarter:before{content:\"\\F243\"}.fa-battery-0:before,.fa-battery-empty:before{content:\"\\F244\"}.fa-mouse-pointer:before{content:\"\\F245\"}.fa-i-cursor:before{content:\"\\F246\"}.fa-object-group:before{content:\"\\F247\"}.fa-object-ungroup:before{content:\"\\F248\"}.fa-sticky-note:before{content:\"\\F249\"}.fa-sticky-note-o:before{content:\"\\F24A\"}.fa-cc-jcb:before{content:\"\\F24B\"}.fa-cc-diners-club:before{content:\"\\F24C\"}.fa-clone:before{content:\"\\F24D\"}.fa-balance-scale:before{content:\"\\F24E\"}.fa-hourglass-o:before{content:\"\\F250\"}.fa-hourglass-1:before,.fa-hourglass-start:before{content:\"\\F251\"}.fa-hourglass-2:before,.fa-hourglass-half:before{content:\"\\F252\"}.fa-hourglass-3:before,.fa-hourglass-end:before{content:\"\\F253\"}.fa-hourglass:before{content:\"\\F254\"}.fa-hand-grab-o:before,.fa-hand-rock-o:before{content:\"\\F255\"}.fa-hand-paper-o:before,.fa-hand-stop-o:before{content:\"\\F256\"}.fa-hand-scissors-o:before{content:\"\\F257\"}.fa-hand-lizard-o:before{content:\"\\F258\"}.fa-hand-spock-o:before{content:\"\\F259\"}.fa-hand-pointer-o:before{content:\"\\F25A\"}.fa-hand-peace-o:before{content:\"\\F25B\"}.fa-trademark:before{content:\"\\F25C\"}.fa-registered:before{content:\"\\F25D\"}.fa-creative-commons:before{content:\"\\F25E\"}.fa-gg:before{content:\"\\F260\"}.fa-gg-circle:before{content:\"\\F261\"}.fa-tripadvisor:before{content:\"\\F262\"}.fa-odnoklassniki:before{content:\"\\F263\"}.fa-odnoklassniki-square:before{content:\"\\F264\"}.fa-get-pocket:before{content:\"\\F265\"}.fa-wikipedia-w:before{content:\"\\F266\"}.fa-safari:before{content:\"\\F267\"}.fa-chrome:before{content:\"\\F268\"}.fa-firefox:before{content:\"\\F269\"}.fa-opera:before{content:\"\\F26A\"}.fa-internet-explorer:before{content:\"\\F26B\"}.fa-television:before,.fa-tv:before{content:\"\\F26C\"}.fa-contao:before{content:\"\\F26D\"}.fa-500px:before{content:\"\\F26E\"}.fa-amazon:before{content:\"\\F270\"}.fa-calendar-plus-o:before{content:\"\\F271\"}.fa-calendar-minus-o:before{content:\"\\F272\"}.fa-calendar-times-o:before{content:\"\\F273\"}.fa-calendar-check-o:before{content:\"\\F274\"}.fa-industry:before{content:\"\\F275\"}.fa-map-pin:before{content:\"\\F276\"}.fa-map-signs:before{content:\"\\F277\"}.fa-map-o:before{content:\"\\F278\"}.fa-map:before{content:\"\\F279\"}.fa-commenting:before{content:\"\\F27A\"}.fa-commenting-o:before{content:\"\\F27B\"}.fa-houzz:before{content:\"\\F27C\"}.fa-vimeo:before{content:\"\\F27D\"}.fa-black-tie:before{content:\"\\F27E\"}.fa-fonticons:before{content:\"\\F280\"}.fa-reddit-alien:before{content:\"\\F281\"}.fa-edge:before{content:\"\\F282\"}.fa-credit-card-alt:before{content:\"\\F283\"}.fa-codiepie:before{content:\"\\F284\"}.fa-modx:before{content:\"\\F285\"}.fa-fort-awesome:before{content:\"\\F286\"}.fa-usb:before{content:\"\\F287\"}.fa-product-hunt:before{content:\"\\F288\"}.fa-mixcloud:before{content:\"\\F289\"}.fa-scribd:before{content:\"\\F28A\"}.fa-pause-circle:before{content:\"\\F28B\"}.fa-pause-circle-o:before{content:\"\\F28C\"}.fa-stop-circle:before{content:\"\\F28D\"}.fa-stop-circle-o:before{content:\"\\F28E\"}.fa-shopping-bag:before{content:\"\\F290\"}.fa-shopping-basket:before{content:\"\\F291\"}.fa-hashtag:before{content:\"\\F292\"}.fa-bluetooth:before{content:\"\\F293\"}.fa-bluetooth-b:before{content:\"\\F294\"}.fa-percent:before{content:\"\\F295\"}.fa-gitlab:before{content:\"\\F296\"}.fa-wpbeginner:before{content:\"\\F297\"}.fa-wpforms:before{content:\"\\F298\"}.fa-envira:before{content:\"\\F299\"}.fa-universal-access:before{content:\"\\F29A\"}.fa-wheelchair-alt:before{content:\"\\F29B\"}.fa-question-circle-o:before{content:\"\\F29C\"}.fa-blind:before{content:\"\\F29D\"}.fa-audio-description:before{content:\"\\F29E\"}.fa-volume-control-phone:before{content:\"\\F2A0\"}.fa-braille:before{content:\"\\F2A1\"}.fa-assistive-listening-systems:before{content:\"\\F2A2\"}.fa-american-sign-language-interpreting:before,.fa-asl-interpreting:before{content:\"\\F2A3\"}.fa-deaf:before,.fa-deafness:before,.fa-hard-of-hearing:before{content:\"\\F2A4\"}.fa-glide:before{content:\"\\F2A5\"}.fa-glide-g:before{content:\"\\F2A6\"}.fa-sign-language:before,.fa-signing:before{content:\"\\F2A7\"}.fa-low-vision:before{content:\"\\F2A8\"}.fa-viadeo:before{content:\"\\F2A9\"}.fa-viadeo-square:before{content:\"\\F2AA\"}.fa-snapchat:before{content:\"\\F2AB\"}.fa-snapchat-ghost:before{content:\"\\F2AC\"}.fa-snapchat-square:before{content:\"\\F2AD\"}.fa-pied-piper:before{content:\"\\F2AE\"}.fa-first-order:before{content:\"\\F2B0\"}.fa-yoast:before{content:\"\\F2B1\"}.fa-themeisle:before{content:\"\\F2B2\"}.fa-google-plus-circle:before,.fa-google-plus-official:before{content:\"\\F2B3\"}.fa-fa:before,.fa-font-awesome:before{content:\"\\F2B4\"}.fa-handshake-o:before{content:\"\\F2B5\"}.fa-envelope-open:before{content:\"\\F2B6\"}.fa-envelope-open-o:before{content:\"\\F2B7\"}.fa-linode:before{content:\"\\F2B8\"}.fa-address-book:before{content:\"\\F2B9\"}.fa-address-book-o:before{content:\"\\F2BA\"}.fa-address-card:before,.fa-vcard:before{content:\"\\F2BB\"}.fa-address-card-o:before,.fa-vcard-o:before{content:\"\\F2BC\"}.fa-user-circle:before{content:\"\\F2BD\"}.fa-user-circle-o:before{content:\"\\F2BE\"}.fa-user-o:before{content:\"\\F2C0\"}.fa-id-badge:before{content:\"\\F2C1\"}.fa-drivers-license:before,.fa-id-card:before{content:\"\\F2C2\"}.fa-drivers-license-o:before,.fa-id-card-o:before{content:\"\\F2C3\"}.fa-quora:before{content:\"\\F2C4\"}.fa-free-code-camp:before{content:\"\\F2C5\"}.fa-telegram:before{content:\"\\F2C6\"}.fa-thermometer-4:before,.fa-thermometer-full:before,.fa-thermometer:before{content:\"\\F2C7\"}.fa-thermometer-3:before,.fa-thermometer-three-quarters:before{content:\"\\F2C8\"}.fa-thermometer-2:before,.fa-thermometer-half:before{content:\"\\F2C9\"}.fa-thermometer-1:before,.fa-thermometer-quarter:before{content:\"\\F2CA\"}.fa-thermometer-0:before,.fa-thermometer-empty:before{content:\"\\F2CB\"}.fa-shower:before{content:\"\\F2CC\"}.fa-bath:before,.fa-bathtub:before,.fa-s15:before{content:\"\\F2CD\"}.fa-podcast:before{content:\"\\F2CE\"}.fa-window-maximize:before{content:\"\\F2D0\"}.fa-window-minimize:before{content:\"\\F2D1\"}.fa-window-restore:before{content:\"\\F2D2\"}.fa-times-rectangle:before,.fa-window-close:before{content:\"\\F2D3\"}.fa-times-rectangle-o:before,.fa-window-close-o:before{content:\"\\F2D4\"}.fa-bandcamp:before{content:\"\\F2D5\"}.fa-grav:before{content:\"\\F2D6\"}.fa-etsy:before{content:\"\\F2D7\"}.fa-imdb:before{content:\"\\F2D8\"}.fa-ravelry:before{content:\"\\F2D9\"}.fa-eercast:before{content:\"\\F2DA\"}.fa-microchip:before{content:\"\\F2DB\"}.fa-snowflake-o:before{content:\"\\F2DC\"}.fa-superpowers:before{content:\"\\F2DD\"}.fa-wpexplorer:before{content:\"\\F2DE\"}.fa-meetup:before{content:\"\\F2E0\"}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0}.sr-only-focusable:active,.sr-only-focusable:focus{position:static;width:auto;height:auto;margin:0;overflow:visible;clip:auto}", ""]);

// exports


/***/ }),

/***/ "../../node_modules/css-loader/index.js!./styles/inconsolata.css":
/***/ (function(module, exports, __webpack_require__) {

var escape = __webpack_require__("../../node_modules/css-loader/lib/url/escape.js");
exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, "@font-face{font-family:Inconsolata;font-style:normal;font-weight:400;src:local(\"Inconsolata Regular\"),local(\"Inconsolata-Regular\"),url(" + escape(__webpack_require__("./styles/fonts/Inconsolata-Regular.ttf")) + ") format(\"truetype\")}@font-face{font-family:Inconsolata;font-style:normal;font-weight:700;src:local(\"Inconsolata Bold\"),local(\"Inconsolata-Bold\"),url(" + escape(__webpack_require__("./styles/fonts/Inconsolata-Bold.ttf")) + ") format(\"truetype\")}", ""]);

// exports


/***/ }),

/***/ "../../node_modules/css-loader/index.js!./styles/neo4j-world.css":
/***/ (function(module, exports, __webpack_require__) {

var escape = __webpack_require__("../../node_modules/css-loader/lib/url/escape.js");
exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, "@font-face{font-family:neo4j-world;src:url(" + escape(__webpack_require__("./styles/fonts/neo4j-world.eot?9li18m")) + ");src:url(" + escape(__webpack_require__("./styles/fonts/neo4j-world.eot?9li18m")) + "#iefix) format(\"embedded-opentype\"),url(" + escape(__webpack_require__("./styles/fonts/neo4j-world.ttf?9li18m")) + ") format(\"truetype\"),url(" + escape(__webpack_require__("./styles/fonts/neo4j-world.woff?9li18m")) + ") format(\"woff\"),url(" + escape(__webpack_require__("./styles/fonts/neo4j-world.svg?9li18m")) + "#neo4j-world) format(\"svg\");font-weight:400;font-style:normal}.nw{font-family:neo4j-world!important;speak:none;font-style:normal;font-weight:400;font-variant:normal;text-transform:none;line-height:1;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}.nw-neo4j-filled-16px .path1:before{content:\"\\1F328\";color:#000}.nw-neo4j-filled-16px .path2:before{content:\"\\1F329\";margin-left:-1em;color:#fff}.nw-neo4j-filled-16px .path3:before{content:\"\\1F32A\";margin-left:-1em;color:#fff}.nw-neo4j-filled-16px .path4:before{content:\"\\1F32B\";margin-left:-1em;color:#fff}.nw-neo4j-filled-16px .path5:before{content:\"\\1F32C\";margin-left:-1em;color:#fff}.nw-neo4j-filled-16px .path6:before{content:\"\\1F32D\";margin-left:-1em;color:#000}.nw-neo4j-filled-16px .path7:before{content:\"\\1F32E\";margin-left:-1em;color:#000}.nw-neo4j-filled-16px .path8:before{content:\"\\1F32F\";margin-left:-1em;color:#000}.nw-neo4j-outline-16px:before{content:\"\\E900\"}.nw-neo4j-filled-32px .path1:before{content:\"\\E909\";color:#000}.nw-neo4j-filled-32px .path2:before{content:\"\\E90A\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path3:before{content:\"\\E90B\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path4:before{content:\"\\E90C\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path5:before{content:\"\\E90D\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path6:before{content:\"\\E90E\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path7:before{content:\"\\E90F\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path8:before{content:\"\\E910\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path9:before{content:\"\\E911\";margin-left:-1em;color:#fff}.nw-neo4j-filled-32px .path10:before{content:\"\\E912\";margin-left:-1em;color:#000}.nw-neo4j-filled-32px .path11:before{content:\"\\E913\";margin-left:-1em;color:#000}.nw-neo4j-filled-32px .path12:before{content:\"\\E914\";margin-left:-1em;color:#000}.nw-neo4j-outline-32px:before{content:\"\\E915\"}.nw-neo4j-filled-64px .path1:before{content:\"\\E916\";color:#000}.nw-neo4j-filled-64px .path2:before{content:\"\\E917\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path3:before{content:\"\\E918\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path4:before{content:\"\\E919\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path5:before{content:\"\\E91A\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path6:before{content:\"\\E91B\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path7:before{content:\"\\E91C\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path8:before{content:\"\\E91D\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path9:before{content:\"\\E91E\";margin-left:-1em;color:#fff}.nw-neo4j-filled-64px .path10:before{content:\"\\E91F\";margin-left:-1em;color:#000}.nw-neo4j-filled-64px .path11:before{content:\"\\E920\";margin-left:-1em;color:#000}.nw-neo4j-filled-64px .path12:before{content:\"\\E921\";margin-left:-1em;color:#000}.nw-neo4j-outline-64px:before{content:\"\\E922\"}", ""]);

// exports


/***/ }),

/***/ "../../node_modules/css-loader/index.js!./styles/open-sans.css":
/***/ (function(module, exports, __webpack_require__) {

var escape = __webpack_require__("../../node_modules/css-loader/lib/url/escape.js");
exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, "@font-face{font-family:Open Sans;font-style:normal;font-weight:300;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-Light.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:italic;font-weight:300;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-LightItalic.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:normal;font-weight:400;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-Regular.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:italic;font-weight:400;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-Italic.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:normal;font-weight:600;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-Semibold.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:italic;font-weight:600;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-SemiboldItalic.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:normal;font-weight:700;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-Bold.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:italic;font-weight:700;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-BoldItalic.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:normal;font-weight:800;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-ExtraBold.ttf")) + ") format(\"truetype\")}@font-face{font-family:Open Sans;font-style:italic;font-weight:800;src:local(\"Open Sans\"),local(\"OpenSans\"),url(" + escape(__webpack_require__("./styles/fonts/OpenSans-ExtraBoldItalic.ttf")) + ") format(\"truetype\")}", ""]);

// exports


/***/ }),

/***/ "../../node_modules/css-loader/index.js!./styles/streamline.css":
/***/ (function(module, exports, __webpack_require__) {

var escape = __webpack_require__("../../node_modules/css-loader/lib/url/escape.js");
exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, "@font-face{font-family:streamline;src:url(" + escape(__webpack_require__("./styles/fonts/streamline.eot")) + ");src:url(" + escape(__webpack_require__("./styles/fonts/streamline.eot")) + "?#iefix) format(\"embedded-opentype\"),url(" + escape(__webpack_require__("./styles/fonts/streamline.woff")) + ") format(\"woff\"),url(" + escape(__webpack_require__("./styles/fonts/streamline.ttf")) + ") format(\"truetype\"),url(" + escape(__webpack_require__("./styles/fonts/streamline.svg")) + "#streamline) format(\"svg\");font-weight:400;font-style:normal}[data-icon]:before{content:attr(data-icon)}[class*=\" sl-\"]:before,[class^=sl-]:before,[data-icon]:before{font-family:streamline!important;font-style:normal!important;font-weight:400!important;font-variant:normal!important;text-transform:none!important;speak:none;line-height:1;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}.sl-star:before{content:\"a\"}.sl-bubble-comment:before{content:\"b\"}.sl-user-circle:before{content:\"c\"}.sl-setting-gear:before{content:\"d\"}.sl-book:before{content:\"e\"}.sl-scale-reduce:before{content:\"g\"}.sl-pin:before{content:\"h\"}.sl-star-circle:before{content:\"s\"}.sl-play:before{content:\"w\"}.sl-cloud:before{content:\"x\"}.sl-alert:before{content:\"y\"}.sl-question-mark:before{content:\"z\"}.sl-setting-adjustment:before{content:\"A\"}.sl-palette:before{content:\"B\"}.sl-business-chart:before{content:\"C\"}.sl-scale-spread:before{content:\"O\"}.sl-cloud-download:before{content:\"P\"}.sl-cloud-upload:before{content:\"Q\"}.sl-database:before{content:\"R\"}.sl-lock-unlock:before{content:\"S\"}.sl-lock:before{content:\"T\"}.sl-heart:before{content:\"i\"}.sl-heart-circle:before{content:\"j\"}.sl-address:before{content:\"k\"}.sl-cloud-delete:before{content:\"m\"}.sl-cloud-checked:before{content:\"n\"}.sl-cloud-sync:before{content:\"o\"}.sl-database-check:before{content:\"p\"}.sl-database-delete:before{content:\"q\"}.sl-window-code:before{content:\"r\"}.sl-arrow-circle-left:before{content:\"v\"}.sl-arrow-circle-right:before{content:\"D\"}.sl-arrow-circle-up:before{content:\"E\"}.sl-arrow-circle-down:before{content:\"F\"}.sl-chevron-left:before{content:\"G\"}.sl-chevron-right:before{content:\"H\"}.sl-chevron-up:before{content:\"J\"}.sl-chevron-down:before{content:\"K\"}.sl-arrow-left:before{content:\"L\"}.sl-arrow-right:before{content:\"M\"}.sl-arrow-up:before{content:\"N\"}.sl-arrow-down:before{content:\"U\"}.sl-database-alert:before{content:\"0\"}.sl-database-block:before{content:\"1\"}.sl-database-lock:before{content:\"2\"}.sl-database-sync:before{content:\"3\"}.sl-database-setting:before{content:\"4\"}.sl-database-edit:before{content:\"5\"}.sl-database-minus:before{content:\"6\"}.sl-database-plus:before{content:\"7\"}.sl-upload-drive:before{content:\"8\"}.sl-download-drive:before{content:\"9\"}.sl-signal-camera:before{content:\"!\"}.sl-user:before{content:'\"'}.sl-pencil:before{content:\"#\"}.sl-bin:before{content:\"$\"}.sl-pencil-circle:before{content:\"%\"}.sl-zoom-in:before{content:\"&\"}.sl-zoom-out:before{content:\"'\"}.sl-upload-arrow:before{content:\"(\"}.sl-download-arrow:before{content:\")\"}.sl-keyhole:before{content:\"*\"}.sl-link:before{content:\"+\"}.sl-link-broken:before{content:\",\"}.sl-show:before{content:\"-\"}.sl-hide:before{content:\".\"}.sl-folder:before{content:\"=\"}.sl-duplicate:before{content:\"l\"}.sl-mail:before{content:\"/\"}.sl-delete:before{content:\"I\"}.sl-double-down:before{content:\"?\"}.sl-double-up:before{content:\"@\"}.sl-triangle-down:before{content:\"V\"}.sl-triangle-up:before{content:\"Y\"}.sl-unfold:before{content:\"Z\"}.sl-chat-bubble:before{content:\">\"}.sl-minus-circle:before{content:\"[\"}.sl-add-circle:before{content:\"]\"}.sl-delete-circle:before{content:\"^\"}.sl-house:before{content:\"_\"}.sl-marker:before{content:\"`\"}.sl-arrange:before{content:\"{\"}.sl-bubble-conversation:before{content:\"\\\\\"}.sl-pulse-graph:before{content:\"\\E000\"}.sl-pencil-filled:before{content:\"t\"}.sl-keyhole-filled:before{content:\"W\"}.sl-star-filled:before{content:\"X\"}.sl-heart-filled:before{content:\"<\"}.sl-play-filled:before{content:\"\\E002\"}.sl-add-filled:before{content:\"\\E003\"}.sl-minus-filled:before{content:\"\\E004\"}.sl-delete-filled:before{content:\"\\E005\"}.sl-thumbnails:before{content:\";\"}.sl-arrow-undo:before{content:\"f\"}.sl-network:before{content:\"u\"}.sl-tag:before{content:\":\"}.sl-arrow-up-right:before{content:\"}\"}.sl-network-trio:before{content:\"~\"}.sl-loop:before{content:\"|\"}.sl-hierarchy:before{content:\"\\E001\"}", ""]);

// exports


/***/ }),

/***/ "../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./components/buttons/style.css":
/***/ (function(module, exports, __webpack_require__) {

exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, ".style__icon___o6PCl{min-width:40px!important;top:4px!important;border:0;background:transparent}.style__tooltip___2z4aB{position:absolute;background-color:grey}.style__action___2M3Gd{padding:0!important;font-size:.8em!important;min-width:50px!important}.style__remove___bSC5d{color:#bcc0c9!important}", ""]);

// exports
exports.locals = {
	"icon": "style__icon___o6PCl",
	"tooltip": "style__tooltip___2z4aB",
	"action": "style__action___2M3Gd",
	"remove": "style__remove___bSC5d"
};

/***/ }),

/***/ "../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./components/icons/style.css":
/***/ (function(module, exports, __webpack_require__) {

exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, ".style__inactive___1HUTh{color:#797979}.style__green___1_JOb,.style__successGreen___3hz1v{color:#4cd950}.style__orange___Bvzy4{color:#f79868}.style__blue___3e0Tl{color:#58c8e3}.style__purple___1KSB9{color:#db7295}.style__alertYellow___1Abks,.style__yellow___3IBJk{color:#ffaf00}.style__warningRed___1omDw{color:#df4d3b}.style__white___1P1DZ{color:#fff}.style__lightBlue___1_ofE{color:#5dade2}@-webkit-keyframes style__neo4j-pulse___2Y6wT{0%,to{color:#1bf621}50%{color:#00a3ff}}@keyframes style__neo4j-pulse___2Y6wT{0%,to{color:#1bf621}50%{color:#00a3ff}}.style__credits___DlncF{-webkit-animation:style__neo4j-pulse___2Y6wT 2s infinite;animation:style__neo4j-pulse___2Y6wT 2s infinite;-webkit-animation-timing-function:ease-in-out;animation-timing-function:ease-in-out}", ""]);

// exports
exports.locals = {
	"inactive": "style__inactive___1HUTh",
	"green": "style__green___1_JOb",
	"successGreen": "style__successGreen___3hz1v",
	"orange": "style__orange___Bvzy4",
	"blue": "style__blue___3e0Tl",
	"purple": "style__purple___1KSB9",
	"yellow": "style__yellow___3IBJk",
	"alertYellow": "style__alertYellow___1Abks",
	"warningRed": "style__warningRed___1omDw",
	"white": "style__white___1P1DZ",
	"lightBlue": "style__lightBlue___1_ofE",
	"credits": "style__credits___DlncF",
	"neo4j-pulse": "style__neo4j-pulse___2Y6wT"
};

/***/ }),

/***/ "../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./modules/Guides/style.css":
/***/ (function(module, exports, __webpack_require__) {

exports = module.exports = __webpack_require__("../../node_modules/css-loader/lib/css-base.js")(false);
// imports


// module
exports.push([module.i, ".carousel,.carousel-container{display:-webkit-box;display:-ms-flexbox;display:flex}.carousel{-webkit-box-orient:vertical;-webkit-box-direction:normal;-ms-flex-direction:column;flex-direction:column}.carousel,.left-button,.right-button{-webkit-box-ordinal-group:2;-ms-flex-order:1;order:1}.left-button,.right-button{top:30px}.slide{overflow-y:auto;font-family:Open Sans,HelveticaNeue-Light,Helvetica Neue Light,Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:200;font-size:14px;font-weight:300;padding-top:2em}.slide .code,.slide code,.slide figure pre,.slide kbd,.slide pre,.slide samp{font-family:Inconsolata,Monaco,Lucida Console,Courier,monospace;background:#f3f3f3;padding-left:.25em;padding-right:.25em;white-space:pre-wrap;word-break:break-all;word-wrap:break-word}.slide th{min-width:50px}.slide pre.code{padding:.5em;margin-top:.5em;margin-bottom:.5em}.slide pre.code.pre-scrollable{max-height:230px}.slide pre.code.runnable{cursor:pointer;border:2px dashed #e1e1e8}.slide pre.code.clicked{border:2px solid #8dd465;opacity:.5}.slide .text-right{text-align:right}.slide .nobreak{word-break:keep-all;white-space:nowrap}.slide .italic{font-style:italic}.slide .light{font-weight:300}.slide .semi-bold{font-weight:600}.slide .bold{font-weight:700}.slide .extra-bold{font-weight:800}.slide .btn--s,.slide .small,.slide small{font-size:.8em}.slide .caps{font-variant:small-caps}.slide .muted{opacity:.7}.slide h1,.slide h2,.slide h3,.slide h4,.slide h5,.slide h6{line-height:.9em;margin:0 0 .25em;font-weight:400}.slide .h1,.slide h1{font-size:2.441em}.slide .h2,.slide h2{font-size:1.953em}.slide .h2.vtop,.slide h2.vtop{margin-top:.2em;margin-bottom:.35em}.slide .h3,.slide h3{font-size:1.563em;font-weight:600}.slide .h3.vtop,.slide h3.vtop{margin-top:.48em;margin-bottom:.4em}.slide .h4,.slide h4{font-size:1.25em}.slide .h4.vtop,.slide h4.vtop{margin-top:.8em;margin-bottom:.55em}.slide .h5,.slide h5{font-size:1em}.slide .h5.vtop,.slide h5.vtop{margin-top:1.2em;margin-bottom:.7em}.slide .h6,.slide h6{font-size:.8em}.slide .h6.vtop,.slide h6.vtop{margin-top:1.75em;margin-bottom:.9em}.slide p{margin-bottom:1em}.slide p.lead{color:#666}.slide ol,.slide ul{list-style-position:outside;padding-left:1em;margin-top:.5em}.slide ul.big{line-height:2em}.slide ul.vtop{margin-top:0}.slide .icon.icon-sm{font-size:.5em}.slide .icon.icon-2x{font-size:2em}.slide .icon.icon-3x{font-size:3em}.slide .icon.icon-4x{font-size:4em}.slide article.help{font-size:15px;color:#666;display:block;width:100%;padding:30px}.slide article.help .main{display:-webkit-box;display:-ms-flexbox;display:flex}.slide .headings,.slide article.help .main{-webkit-box-flex:0;-ms-flex:0 0 25%;flex:0 0 25%}.slide .headings .title{font-size:24px;font-weight:400;color:#333}.slide .content p{margin-bottom:10px}.slide .content img{max-width:900px;max-height:300px}.slide .example{padding-left:25%}.slide .runnable pre{cursor:pointer;border:2px dashed #e1e1e8;padding-left:1em;padding-right:1em}.slide .links{display:table}.slide .links .link{display:table-row}.slide .links .link .content,.slide .links .link .title{display:table-cell;padding:5px;font-size:13px}.slide .links .link .title{text-align:right;font-weight:700}.slide [exec-topic],.slide [play-topic],.slide [server-topic],.slide a[help-topic]{background-color:#f8f8f8;border-radius:3px;border:1px solid #dadada;display:inline-block;font-family:Monaco,Courier New,Terminal,monospace;font-size:12px;line-height:18px;margin-bottom:5px;margin-right:5px;padding:0 4px;color:#428bca;cursor:pointer;text-decoration:none}.slide pre{color:#333;word-break:break-all;word-wrap:break-word;background-color:#f5f5f5;padding:10px}.slide code{color:#c7254e;background-color:#f9f2f4;border-radius:4px}.slide figcaption{font-style:italic;font-size:13px;text-align:center}.slide footer{text-align:center}.slide .btn-cta{padding:1em;background-color:#008cc1;color:#fff;border-radius:3px;border:1px solid #dadada;display:inline-block}.slide .teaser{margin:.5em;padding:10px;border:1px solid #ccc;border-radius:4px;height:270px;float:left;position:relative}.slide .teaser.teaser-2{width:45%;min-width:285px;height:320px;overflow:hidden}.slide .teaser.teaser-2 .icon-holder{max-height:160px;overflow-y:auto}.slide .teaser.teaser-2 .icon{width:12%}.slide .teaser.teaser-2 .topic-bullets{max-width:80%}.slide .teaser.teaser-3{width:30%;min-width:190px;overflow:hidden}.slide .teaser button{font-weight:600;position:absolute;bottom:10px;width:90%;left:5%;outline:none}.slide .teaser .icon{float:left;max-width:67px;width:25%}.slide .teaser .icon.sl{padding-left:10px;font-size:48px}.slide .teaser .icon.sl.green{color:#65b144}.slide .teaser .icon.sl.yellow{color:#fdcc59}.slide .teaser .icon.sl.red{color:#ff5641}.slide .teaser .topic-bullets{float:left;word-wrap:break-word;min-width:100px;max-width:65%;list-style:none;padding-left:10px;margin-top:0}.slide .teaser .topic-bullets :first-child{margin-top:0}.slide .teaser .topic-bullets li{margin-top:5px}.slide .teaser .icon-holder{margin-top:10px;max-height:130px;overflow:hidden}", ""]);

// exports
exports.locals = {
	"carousel-container": "carousel-container",
	"carouselContainer": "carousel-container",
	"carousel": "carousel",
	"left-button": "left-button",
	"leftButton": "left-button",
	"right-button": "right-button",
	"rightButton": "right-button",
	"slide": "slide",
	"code": "code",
	"pre-scrollable": "pre-scrollable",
	"preScrollable": "pre-scrollable",
	"runnable": "runnable",
	"clicked": "clicked",
	"text-right": "text-right",
	"textRight": "text-right",
	"nobreak": "nobreak",
	"italic": "italic",
	"light": "light",
	"semi-bold": "semi-bold",
	"semiBold": "semi-bold",
	"bold": "bold",
	"extra-bold": "extra-bold",
	"extraBold": "extra-bold",
	"small": "small",
	"btn--s": "btn--s",
	"btnS": "btn--s",
	"caps": "caps",
	"muted": "muted",
	"h1": "h1",
	"h2": "h2",
	"vtop": "vtop",
	"h3": "h3",
	"h4": "h4",
	"h5": "h5",
	"h6": "h6",
	"lead": "lead",
	"big": "big",
	"icon": "icon",
	"icon-sm": "icon-sm",
	"iconSm": "icon-sm",
	"icon-2x": "icon-2x",
	"icon2X": "icon-2x",
	"icon-3x": "icon-3x",
	"icon3X": "icon-3x",
	"icon-4x": "icon-4x",
	"icon4X": "icon-4x",
	"help": "help",
	"main": "main",
	"headings": "headings",
	"title": "title",
	"content": "content",
	"example": "example",
	"links": "links",
	"link": "link",
	"btn-cta": "btn-cta",
	"btnCta": "btn-cta",
	"teaser": "teaser",
	"teaser-2": "teaser-2",
	"teaser2": "teaser-2",
	"icon-holder": "icon-holder",
	"iconHolder": "icon-holder",
	"topic-bullets": "topic-bullets",
	"topicBullets": "topic-bullets",
	"teaser-3": "teaser-3",
	"teaser3": "teaser-3",
	"sl": "sl",
	"green": "green",
	"yellow": "yellow",
	"red": "red"
};

/***/ }),

/***/ "./components/buttons/style.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./components/buttons/style.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./components/icons/style.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./components/icons/style.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./index.jsx":
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
// ESM COMPAT FLAG
__webpack_require__.r(__webpack_exports__);

// NAMESPACE OBJECT: ./styles/themes.js
var themes_namespaceObject = {};
__webpack_require__.r(themes_namespaceObject);
__webpack_require__.d(themes_namespaceObject, "base", function() { return base; });
__webpack_require__.d(themes_namespaceObject, "normal", function() { return normal; });
__webpack_require__.d(themes_namespaceObject, "outline", function() { return outline; });
__webpack_require__.d(themes_namespaceObject, "dark", function() { return dark; });

// NAMESPACE OBJECT: ../shared/services/exceptionMessages.js
var exceptionMessages_namespaceObject = {};
__webpack_require__.r(exceptionMessages_namespaceObject);
__webpack_require__.d(exceptionMessages_namespaceObject, "AddServerValidationError", function() { return AddServerValidationError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "CreateDataSourceValidationError", function() { return CreateDataSourceValidationError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "RemoveDataSourceValidationError", function() { return RemoveDataSourceValidationError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "BoltConnectionError", function() { return BoltConnectionError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "BoltError", function() { return BoltError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "Neo4jError", function() { return Neo4jError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "UnknownCommandError", function() { return UnknownCommandError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "BookmarkNotFoundError", function() { return BookmarkNotFoundError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "OpenConnectionNotFoundError", function() { return OpenConnectionNotFoundError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "CouldNotFetchRemoteGuideError", function() { return CouldNotFetchRemoteGuideError; });
__webpack_require__.d(exceptionMessages_namespaceObject, "FetchURLError", function() { return FetchURLError; });

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/react/index.js
var react = __webpack_require__("../../node_modules/react/index.js");
var react_default = /*#__PURE__*/__webpack_require__.n(react);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/react-dom/index.js
var react_dom = __webpack_require__("../../node_modules/react-dom/index.js");
var react_dom_default = /*#__PURE__*/__webpack_require__.n(react_dom);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@babel/polyfill/lib/index.js
var lib = __webpack_require__("../../node_modules/@babel/polyfill/lib/index.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/raf/polyfill.js
var polyfill = __webpack_require__("../../node_modules/raf/polyfill.js");

// EXTERNAL MODULE: ./styles/streamline.css
var streamline = __webpack_require__("./styles/streamline.css");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/styled-components/dist/styled-components.es.js
var styled_components_es = __webpack_require__("../../node_modules/styled-components/dist/styled-components.es.js");

// CONCATENATED MODULE: ./styles/global-styles.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


Object(styled_components_es["injectGlobal"])([""]);
// EXTERNAL MODULE: ./styles/neo4j-world.css
var neo4j_world = __webpack_require__("./styles/neo4j-world.css");

// EXTERNAL MODULE: ./styles/font-awesome.min.css
var font_awesome_min = __webpack_require__("./styles/font-awesome.min.css");

// EXTERNAL MODULE: ./styles/inconsolata.css
var inconsolata = __webpack_require__("./styles/inconsolata.css");

// EXTERNAL MODULE: ./styles/open-sans.css
var open_sans = __webpack_require__("./styles/open-sans.css");

// CONCATENATED MODULE: ./init.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */









// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/redux-observable/lib/es/index.js + 4 modules
var es = __webpack_require__("../../node_modules/redux-observable/lib/es/index.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/redux/es/index.js + 15 modules
var redux_es = __webpack_require__("../../node_modules/redux/es/index.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/react-redux/es/index.js + 20 modules
var react_redux_es = __webpack_require__("../../node_modules/react-redux/es/index.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/suber/lib/bus.js
var bus = __webpack_require__("../../node_modules/suber/lib/bus.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/react-suber/lib/index.js
var react_suber_lib = __webpack_require__("../../node_modules/react-suber/lib/index.js");

// CONCATENATED MODULE: ../shared/modules/settings/settingsDuck.js
function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
var NAME = 'settings';
var UPDATE = 'settings/UPDATE';
var REPLACE = 'settings/REPLACE';
var getSettings = function getSettings(state) {
  return state[NAME];
};
var getInitCmd = function getInitCmd(state) {
  return state[NAME].initCmd || initialState.initCmd;
};
var getTheme = function getTheme(state) {
  return state[NAME].theme || initialState.theme;
};
var getScrollToTop = function getScrollToTop(state) {
  return state[NAME].scrollToTop;
};
var getCmdChar = function getCmdChar(state) {
  return state[NAME].cmdchar || initialState.cmdchar;
};
var initialState = {
  cmdchar: ':',
  maxHistory: 30,
  theme: 'normal',
  useBoltRouting: false,
  initCmd: ':play start',
  scrollToTop: true,
  maxFrames: 30
};
function settings() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : initialState;
  var action = arguments.length > 1 ? arguments[1] : undefined;
  switch (action.type) {
    default:
      return _objectSpread(_objectSpread({}, initialState), state);
  }
}
var update = function update(settings) {
  return {
    type: UPDATE,
    state: settings
  };
};
var replace = function replace(settings) {
  return {
    type: REPLACE,
    state: settings
  };
};
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/uuid/index.js
var uuid = __webpack_require__("../../node_modules/uuid/index.js");
var uuid_default = /*#__PURE__*/__webpack_require__.n(uuid);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/rxjs/add/operator/do.js
var operator_do = __webpack_require__("../../node_modules/rxjs/add/operator/do.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/rxjs/add/operator/mapTo.js
var mapTo = __webpack_require__("../../node_modules/rxjs/add/operator/mapTo.js");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/url-parse/index.js
var url_parse = __webpack_require__("../../node_modules/url-parse/index.js");
var url_parse_default = /*#__PURE__*/__webpack_require__.n(url_parse);

// CONCATENATED MODULE: ../shared/services/utils.js
function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var deepEquals = function deepEquals(x, y) {
  if (x && y && _typeof(x) === 'object' && _typeof(y) === 'object') {
    if (Object.keys(x).length !== Object.keys(y).length) return false;
    return Object.keys(x).every(function (key) {
      return deepEquals(x[key], y[key]);
    });
  }
  return x === y;
};
var shallowEquals = function shallowEquals(a, b) {
  for (var key in a) {
    if (a[key] !== b[key]) return false;
  }
  for (var _key in b) {
    if (!(_key in a)) return false;
  }
  return true;
};
var flatten = function flatten(arr) {
  return arr.reduce(function (a, b) {
    return a.concat(Array.isArray(b) ? flatten(b) : b);
  }, []);
};
var moveInArray = function moveInArray(fromIndex, toIndex, arr) {
  if (!Array.isArray(arr)) return false;
  if (fromIndex < 0 || fromIndex >= arr.length) return false;
  if (toIndex < 0 || toIndex >= arr.length) return false;
  var newArr = [].concat(arr);
  var el = arr[fromIndex];
  newArr.splice(fromIndex, 1);
  newArr.splice(toIndex, 0, el);
  return newArr;
};
var debounce = function debounce(fn, time) {
  var context = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  var pending;
  return function () {
    for (var _len = arguments.length, args = new Array(_len), _key2 = 0; _key2 < _len; _key2++) {
      args[_key2] = arguments[_key2];
    }
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () {
      return typeof fn === 'function' && fn.apply(context, args);
    }, parseInt(time));
  };
};
var throttle = function throttle(fn, time) {
  var context = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  var blocking;
  return function () {
    if (blocking) return;
    blocking = true;
    for (var _len2 = arguments.length, args = new Array(_len2), _key3 = 0; _key3 < _len2; _key3++) {
      args[_key3] = arguments[_key3];
    }
    typeof fn === 'function' && fn.apply(context, args);
    setTimeout(function () {
      return blocking = false;
    }, parseInt(time));
  };
};
var firstSuccessPromise = function firstSuccessPromise(list, fn) {
  return list.reduce(function (promise, item) {
    return promise["catch"](function () {
      return fn(item);
    }).then(function (r) {
      return Promise.resolve(r);
    });
  }, Promise.reject(new Error()));
};
var isRoutingHost = function isRoutingHost(host) {
  return /^bolt\+routing:\/\//.test(host);
};
var hostIsAllowed = function hostIsAllowed(uri) {
  var whitelist = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
  if (whitelist === '*') return true;
  var urlInfo = utils_getUrlInfo(uri);
  var hostname = urlInfo.hostname;
  var hostnamePlusProtocol = urlInfo.protocol + '//' + hostname;
  var whitelistedHosts = whitelist && whitelist !== '' ? extractWhitelistFromConfigString(whitelist) : [];
  return whitelistedHosts.indexOf(hostname) > -1 || whitelistedHosts.indexOf(hostnamePlusProtocol) > -1;
};
var extractWhitelistFromConfigString = function extractWhitelistFromConfigString(str) {
  return str.split(',').map(function (s) {
    return s.trim().replace(/\/$/, '');
  });
};
var addProtocolsToUrlList = function addProtocolsToUrlList(list) {
  return list.reduce(function (all, uri) {
    if (!uri || uri === '*') return all;
    var urlInfo = utils_getUrlInfo(uri);
    if (urlInfo.protocol) return all.concat(uri);
    return all.concat(['https://' + uri, 'http://' + uri]);
  }, []);
};
var utils_getUrlInfo = function getUrlInfo(url) {
  var protocolMissing = false;

  // prepend a default protocol, if none was found
  if (url.match(/^(.+:\/\/)?/)[1] === undefined) {
    url = 'http://' + url;
    protocolMissing = true;
  }
  var _parseUrl = url_parse_default()(url, {}),
    protocol = _parseUrl.protocol,
    username = _parseUrl.username,
    password = _parseUrl.password,
    host = _parseUrl.host,
    hostname = _parseUrl.hostname,
    port = _parseUrl.port,
    pathname = _parseUrl.pathname,
    search = _parseUrl.query,
    hash = _parseUrl.hash;
  return {
    protocol: protocolMissing ? '' : protocol,
    username: username,
    password: password,
    host: host,
    hostname: hostname,
    port: port,
    pathname: pathname,
    search: search,
    hash: hash
  };
};
var getUrlParamValue = function getUrlParamValue(name, url) {
  if (!url) return false;
  var out = [];
  var re = new RegExp('[\\?&]' + name + '=([^&#]*)', 'g');
  var results;
  while ((results = re.exec(url)) !== null) {
    if (results && results[1]) out.push(results[1]);
  }
  if (!out.length) return undefined;
  return out;
};
var toHumanReadableBytes = function toHumanReadableBytes(input) {
  var number = +input;
  if (!isFinite(number)) {
    return '-';
  }
  if (number < 1024) {
    return "".concat(number, " B");
  }
  number /= 1024;
  var units = ['KiB', 'MiB', 'GiB', 'TiB'];
  for (var _i = 0, _Array$from = Array.from(units); _i < _Array$from.length; _i++) {
    var unit = _Array$from[_i];
    if (number < 1024) {
      return "".concat(number.toFixed(2), " ").concat(unit);
    }
    number /= 1024;
  }
  return "".concat(number.toFixed(2), " PiB");
};
var getBrowserName = function getBrowserName() {
  if (!!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0) {
    return 'Opera';
  }
  if (typeof InstallTrigger !== 'undefined') {
    return 'Firefox';
  }
  if (navigator.userAgent.match(/Version\/[\d.]+.*Safari/)) {
    return 'Safari';
  }
  if (window.chrome) {
    return 'Chrome';
  }
  if (document.documentMode) {
    return 'Internet Explorer';
  }
  if (window.StyleMedia) {
    return 'Edge';
  }
  return 'Unknown';
};
var removeComments = function removeComments() {
  var string = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : '';
  return string.split(/\r?\n/).filter(function (line) {
    return !line.startsWith('//');
  }).join('\r\n');
};
var canUseDOM = function canUseDOM() {
  return !!(typeof window !== 'undefined' && window.document && window.document.createElement);
};
var ecsapeCypherMetaItem = function ecsapeCypherMetaItem(str) {
  return /^[A-Za-z][A-Za-z0-9_]*$/.test(str) ? str : '`' + str.replace(/`/g, '``') + '`';
};
var parseTimeMillis = function parseTimeMillis(timeWithOrWithoutUnit) {
  timeWithOrWithoutUnit += ''; // cast to string
  var readUnit = timeWithOrWithoutUnit.match(/\D+/);
  var value = parseInt(timeWithOrWithoutUnit);
  var unit = readUnit === undefined || readUnit === null ? 's' : readUnit[0]; // Assume seconds

  switch (unit) {
    case 'ms':
      return value;
    case 's':
      return value * 1000;
    case 'm':
      return value * 60 * 1000;
    default:
      return 0;
  }
};
var stringifyMod = function stringifyMod(value) {
  var modFn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
  var prettyLevel = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  var skipOpeningIndentation = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
  prettyLevel = !prettyLevel ? false : prettyLevel === true ? 1 : parseInt(prettyLevel);
  var nextPrettyLevel = prettyLevel ? prettyLevel + 1 : false;
  var newLine = prettyLevel ? '\n' : '';
  var indentation = prettyLevel && !skipOpeningIndentation ? Array(prettyLevel).join('  ') : '';
  var endIndentation = prettyLevel ? Array(prettyLevel).join('  ') : '';
  var propSpacing = prettyLevel ? ' ' : '';
  var toString = Object.prototype.toString;
  var isArray = Array.isArray || function (a) {
    return toString.call(a) === '[object Array]';
  };
  var escMap = {
    '"': '\\"',
    '\\': '\\\\',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t'
  };
  var escFunc = function escFunc(m) {
    return escMap[m] || "\\u" + (m.charCodeAt(0) + 0x10000).toString(16).substr(1);
  };
  var escRE = /[\\"\u0000-\u001F\u2028\u2029]/g;
  if (modFn) {
    var modVal = modFn && modFn(value);
    if (typeof modVal !== 'undefined') return indentation + modVal;
  }
  if (value == null) return indentation + 'null';
  if (typeof value === 'number') {
    return indentation + (isFinite(value) ? value.toString() : 'null');
  }
  if (typeof value === 'boolean') return indentation + value.toString();
  if (_typeof(value) === 'object') {
    if (typeof value.toJSON === 'function') {
      return stringifyMod(value.toJSON(), modFn, nextPrettyLevel);
    } else if (isArray(value)) {
      var hasValues = false;
      var res = '';
      for (var i = 0; i < value.length; i++) {
        hasValues = true;
        res += (i ? ',' : '') + newLine + stringifyMod(value[i], modFn, nextPrettyLevel);
      }
      return indentation + '[' + res + (hasValues ? newLine + endIndentation : '') + ']';
    } else if (toString.call(value) === '[object Object]') {
      var tmp = [];
      for (var k in value) {
        if (value.hasOwnProperty(k)) {
          tmp.push(stringifyMod(k, modFn, nextPrettyLevel) + ':' + propSpacing + stringifyMod(value[k], modFn, nextPrettyLevel, true));
        }
      }
      return indentation + '{' + newLine + tmp.join(',' + newLine) + newLine + endIndentation + '}';
    }
  }
  return indentation + '"' + value.toString().replace(escRE, escFunc) + '"';
};

// Epic helpers
var put = function put(dispatch) {
  return function (action) {
    return dispatch(action);
  };
};
// CONCATENATED MODULE: ../shared/modules/app/appDuck.js
function appDuck_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function appDuck_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? appDuck_ownKeys(Object(source), !0).forEach(function (key) { appDuck_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : appDuck_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function appDuck_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
var appDuck_NAME = 'app';
var APP_START = "".concat(appDuck_NAME, "/APP_START");
var USER_CLEAR = "".concat(appDuck_NAME, "/USER_CLEAR");

// Selectors

// Reducer
function reducer() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
  var action = arguments.length > 1 ? arguments[1] : undefined;
  switch (action.type) {
    case APP_START:
      return appDuck_objectSpread({}, state);
    default:
      return state;
  }
}
// CONCATENATED MODULE: ../shared/modules/stream/streamDuck.js
function streamDuck_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function streamDuck_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? streamDuck_ownKeys(Object(source), !0).forEach(function (key) { streamDuck_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : streamDuck_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function streamDuck_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */







var JAVA_PERMISSION_ERROR_SHOW = "JAVA_PERMISSION_ERROR_SHOW";
var JAVA_PERMISSION_ERROR_HIDE = "JAVA_PERMISSION_ERROR_HIDE";
var streamDuck_NAME = "frames";
var ADD = "frames/ADD";
var REMOVE = "frames/REMOVE";
var CLEAR_ALL = "frames/CLEAR_ALL";
var FRAME_TYPE_FILTER_UPDATED = "frames/FRAME_TYPE_FILTER_UPDATED";
var PIN = "".concat(streamDuck_NAME, "/PIN");
var UNPIN = "".concat(streamDuck_NAME, "/UNPIN");
var SET_RECENT_VIEW = "frames/SET_RECENT_VIEW";
var SET_MAX_FRAMES = streamDuck_NAME + "/SET_MAX_FRAMES";

/**
 * Selectors
*/
function getFrames(state) {
  return state[streamDuck_NAME].allIds.map(function (id) {
    return state[streamDuck_NAME].byId[id];
  });
}
function getFramesInContext(state, context) {
  return getFrames(state).filter(function (f) {
    return f.context === context;
  });
}
function getRecentView(state) {
  return state[streamDuck_NAME].recentView;
}

/**
 * Reducer helpers
*/
function addFrame(state, newState) {
  var byId = Object.assign({}, state.byId, streamDuck_defineProperty({}, newState.id, newState));
  var allIds = [].concat(state.allIds);
  if (allIds.indexOf(newState.id) < 0) {
    // new frame
    var pos = findFirstFreePos(state);
    allIds.splice(pos, 0, newState.id);
  }
  return ensureFrameLimit(streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
    allIds: allIds,
    byId: byId
  }));
}
function removeFrame(state, id) {
  var byId = Object.assign({}, state.byId);
  delete byId[id];
  var allIds = state.allIds.filter(function (fid) {
    return fid !== id;
  });
  return Object.assign({}, state, {
    allIds: allIds,
    byId: byId
  });
}
function pinFrame(state, id) {
  var pos = state.allIds.indexOf(id);
  var allIds = moveInArray(pos, 0, state.allIds); // immutable operation
  var byId = Object.assign({}, state.byId);
  byId[id].isPinned = true;
  return streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
    allIds: allIds,
    byId: byId
  });
}
function unpinFrame(state, id) {
  var currentPos = state.allIds.indexOf(id);
  var pos = findFirstFreePos(state);
  var allIds = moveInArray(currentPos, pos - 1, state.allIds); // immutable operation
  var byId = Object.assign({}, state.byId);
  byId[id].isPinned = false;
  return streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
    allIds: allIds,
    byId: byId
  });
}
function findFirstFreePos(_ref) {
  var byId = _ref.byId,
    allIds = _ref.allIds;
  var freePos = -1;
  allIds.forEach(function (id, index) {
    if (freePos > -1 || byId[id].isPinned) return;
    freePos = index;
  });
  return freePos === -1 ? allIds.length : freePos;
}
function setRecentViewHelper(state, recentView) {
  return Object.assign({}, state, {
    recentView: recentView
  });
}
function ensureFrameLimit(state) {
  var limit = state.maxFrames || 1;
  if (state.allIds.length <= limit) return state;
  var numToRemove = state.allIds.length - limit;
  var removeIds = state.allIds.slice(-1 * numToRemove).filter(function (id) {
    return !state.byId[id].isPinned;
  });
  var byId = streamDuck_objectSpread({}, state.byId);
  removeIds.forEach(function (id) {
    return delete byId[id];
  });
  return streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
    allIds: state.allIds.slice(0, state.allIds.length - removeIds.length),
    byId: byId
  });
}

/** Inital state */
var streamDuck_initialState = {
  allIds: [],
  byId: {},
  recentView: null,
  maxFrames: 30,
  javaPermErrorShow: false
};

/**
 * Reducer
*/
function streamDuck_reducer() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : streamDuck_initialState;
  var action = arguments.length > 1 ? arguments[1] : undefined;
  switch (action.type) {
    case APP_START:
      // Handle default
      return addFrame(state, {
        id: "MAPPING",
        type: "mapping"
      });
    case ADD:
      return addFrame(state, action.state);
    case REMOVE:
      return removeFrame(state, action.id);
    case CLEAR_ALL:
      return streamDuck_objectSpread({}, streamDuck_initialState);
    case PIN:
      return pinFrame(state, action.id);
    case UNPIN:
      return unpinFrame(state, action.id);
    case SET_RECENT_VIEW:
      return setRecentViewHelper(state, action.view);
    case SET_MAX_FRAMES:
      var newState = streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
        maxFrames: action.maxFrames
      });
      return ensureFrameLimit(newState);
    case JAVA_PERMISSION_ERROR_SHOW:
      return streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
        javaPermErrorShow: true
      });
    case JAVA_PERMISSION_ERROR_HIDE:
      return streamDuck_objectSpread(streamDuck_objectSpread({}, state), {}, {
        javaPermErrorShow: false
      });
    default:
      return state;
  }
}

// Action creators
function add(payload) {
  return {
    type: ADD,
    state: Object.assign({}, payload, {
      id: payload.id || uuid_default.a.v1()
    })
  };
}
function remove(id) {
  return {
    type: REMOVE,
    id: id
  };
}
function clear() {
  return {
    type: CLEAR_ALL
  };
}
function pin(id) {
  return {
    type: PIN,
    id: id
  };
}
function unpin(id) {
  return {
    type: UNPIN,
    id: id
  };
}
function setRecentView(view) {
  return {
    type: SET_RECENT_VIEW,
    view: view
  };
}

// Epics
var streamDuck_maxFramesConfigEpic = function maxFramesConfigEpic(action$, store) {
  return action$.ofType(UPDATE)["do"](function (action) {
    var newMaxFrames = action.state.maxFrames;
    if (!newMaxFrames) return;
    store.dispatch({
      type: SET_MAX_FRAMES,
      maxFrames: newMaxFrames
    });
  }).mapTo({
    type: "NOOP"
  });
};
// CONCATENATED MODULE: ../shared/modules/commands/commandsDuck.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
var commandsDuck_NAME = 'commands';
var USER_COMMAND_QUEUED = commandsDuck_NAME + '/USER_COMMAND_QUEUED';
var SYSTEM_COMMAND_QUEUED = commandsDuck_NAME + '/SYSTEM_COMMAND_QUEUED';
var UNKNOWN_COMMAND = commandsDuck_NAME + '/UNKNOWN_COMMAND';
var KNOWN_COMMAND = commandsDuck_NAME + '/KNOWN_COMMAND';
var SHOW_ERROR_MESSAGE = commandsDuck_NAME + '/SHOW_ERROR_MESSAGE';
var SHOW_INFO_MESSAGE = commandsDuck_NAME + '/SHOW_INFO_MESSAGE';
var CYPHER = commandsDuck_NAME + '/CYPHER';
var CYPHER_SUCCEEDED = commandsDuck_NAME + '/CYPHER_SUCCEEDED';
var CYPHER_FAILED = commandsDuck_NAME + '/CYPHER_FAILED';
var FETCH_GUIDE_FROM_WHITELIST = commandsDuck_NAME + 'FETCH_GUIDE_FROM_WHITELIST';
var commandsDuck_initialState = {
  lastCommandWasUnknown: false
};
var wasUnknownCommand = function wasUnknownCommand(state) {
  return state[commandsDuck_NAME].lastCommandWasUnknown || commandsDuck_initialState.lastCommandWasUnknown;
};
var getErrorMessage = function getErrorMessage(state) {
  return state[commandsDuck_NAME].errorMessage;
};
var getInfoMessage = function getInfoMessage(state) {
  return state[commandsDuck_NAME].infoMessage;
};
function commandsDuck_reducer() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : commandsDuck_initialState;
  var action = arguments.length > 1 ? arguments[1] : undefined;
  switch (action.type) {
    case UNKNOWN_COMMAND:
      return {
        lastCommandWasUnknown: true
      };
    case KNOWN_COMMAND:
      return {
        lastCommandWasUnknown: false
      };
    case SHOW_ERROR_MESSAGE:
      return {
        lastCommandWasUnknown: false,
        errorMessage: action.errorMessage
      };
    case SHOW_INFO_MESSAGE:
      return {
        lastCommandWasUnknown: false,
        infoMessage: action.infoMessage
      };
    default:
      return state;
  }
}

// Action creators
var executeCommand = function executeCommand(cmd, contextId) {
  var requestId = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  return {
    type: USER_COMMAND_QUEUED,
    cmd: cmd,
    id: contextId,
    requestId: requestId
  };
};
var executeSystemCommand = function executeSystemCommand(cmd, contextId) {
  var requestId = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  return {
    type: SYSTEM_COMMAND_QUEUED,
    cmd: cmd,
    id: contextId,
    requestId: requestId
  };
};
var unknownCommand = function unknownCommand(cmd) {
  return {
    type: UNKNOWN_COMMAND,
    cmd: cmd
  };
};
var showErrorMessage = function showErrorMessage(errorMessage) {
  return {
    type: SHOW_ERROR_MESSAGE,
    errorMessage: errorMessage
  };
};
var showInfoMessage = function showInfoMessage(infoMessage) {
  return {
    type: SHOW_INFO_MESSAGE,
    infoMessage: infoMessage
  };
};
var cypher = function cypher(query) {
  return {
    type: CYPHER,
    query: query
  };
};
var successfulCypher = function successfulCypher(query) {
  return {
    type: CYPHER_SUCCEEDED,
    query: query
  };
};
var unsuccessfulCypher = function unsuccessfulCypher(query) {
  return {
    type: CYPHER_FAILED,
    query: query
  };
};
var fetchGuideFromWhitelistAction = function fetchGuideFromWhitelistAction(url) {
  return {
    type: FETCH_GUIDE_FROM_WHITELIST,
    url: url
  };
};
// CONCATENATED MODULE: ../shared/modules/mapping/mappingDuck.js
function mappingDuck_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function mappingDuck_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? mappingDuck_ownKeys(Object(source), !0).forEach(function (key) { mappingDuck_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : mappingDuck_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function mappingDuck_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
var mappingDuck_NAME = 'mapping';
var NODE_CLICK = "".concat(mappingDuck_NAME, "/NODE_CLICK");
var EDGE_CLICK = "".concat(mappingDuck_NAME, "/EDGE_CLICK");
var LABEL_CHANGED = "".concat(mappingDuck_NAME, "/LABEL_CHANGED");
var TYPE_CHANGED = "".concat(mappingDuck_NAME, "/TYPE_CHANGED");
// For the first frame
var MAPPING_DONE = "".concat(mappingDuck_NAME, "/MAPPING_DONE");
var CONNECTION_SAVED = "".concat(mappingDuck_NAME, "/CONNECTION_SAVED");
var CONNECTION_UPDATE = "".concat(mappingDuck_NAME, "/CONNECTION_UPDATE");
var RESTART_WIZARD = "".concat(mappingDuck_NAME, "/RESTART_WIZARD");
var ADD_ITEM = "".concat(mappingDuck_NAME, "/ADD_ITEM");
var SKIP_ITEM = "".concat(mappingDuck_NAME, "/SKIP_ITEM");
var mappingDuck_initialState = {
  nodeKey: '',
  edgeKey: '',
  nodeIndex: -1,
  edgeIndex: -1,
  nodes: [],
  relationships: [],
  restartWizard: false
};
function mappingDuck_reducer() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : mappingDuck_initialState;
  var action = arguments.length > 1 ? arguments[1] : undefined;
  switch (action.type) {
    case SKIP_ITEM:
      return mappingDuck_objectSpread(mappingDuck_objectSpread({}, state), {}, {
        nodes: action.nodes,
        relationships: action.relationships,
        mappings: action.mappings
      });
    case ADD_ITEM:
      return mappingDuck_objectSpread(mappingDuck_objectSpread({}, state), {}, {
        nodes: action.nodes,
        relationships: action.relationships,
        mappings: action.mappings
      });
    case NODE_CLICK:
      return Object.assign({}, mappingDuck_initialState, {
        nodeKey: action.nodeKey,
        nodeIndex: action.index,
        nodes: action.nodes,
        relationships: action.relationships,
        connectionName: action.connectionName,
        mappings: action.mappings
      });
    case EDGE_CLICK:
      return Object.assign({}, mappingDuck_initialState, {
        edgeKey: action.edgeKey,
        edgeIndex: action.index,
        nodes: action.nodes,
        relationships: action.relationships,
        connectionName: action.connectionName,
        mappings: action.mappings
      });
    case LABEL_CHANGED:
      return Object.assign({}, mappingDuck_initialState, {
        value: action.value,
        node: action.node,
        nodes: action.nodes,
        relationships: action.relationships,
        connectionName: action.connectionName,
        mappings: action.mappings
      });
    case TYPE_CHANGED:
      return Object.assign({}, mappingDuck_initialState, {
        value: action.value,
        edge: action.edge,
        nodes: action.nodes,
        relationships: action.relationships,
        connectionName: action.connectionName,
        mappings: action.mappings
      });
    case CONNECTION_SAVED:
      return Object.assign({}, state, {
        connectionName: action.connectionName
      });
    case CONNECTION_UPDATE:
      return Object.assign({}, state, {
        connectionInfo: action.connectionInfo
      });
    case MAPPING_DONE:
      return Object.assign({}, state, {
        nodes: action.nodes,
        relationships: action.relationships,
        mappings: action.mappings,
        connectionName: action.connectionName
      });
    case RESTART_WIZARD:
      return Object.assign({}, state, {
        restartWizard: action.restartWizard
      });
    default:
      return state;
  }
}
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/uuid/v1.js
var v1 = __webpack_require__("../../node_modules/uuid/v1.js");
var v1_default = /*#__PURE__*/__webpack_require__.n(v1);

// CONCATENATED MODULE: ./services/ConnectionStorage.js
function ConnectionStorage_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function ConnectionStorage_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ConnectionStorage_ownKeys(Object(source), !0).forEach(function (key) { ConnectionStorage_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ConnectionStorage_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function ConnectionStorage_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
/**
 * Manage data source connections for a single project
 * It saves into localStorage
 */


var ConnectionStorage_ConnectionStorage = /*#__PURE__*/function () {
  function ConnectionStorage(projectId) {
    _classCallCheck(this, ConnectionStorage);
    this.projectId = projectId || 'default';
    this.currentConnection = {};
  }
  _createClass(ConnectionStorage, [{
    key: "buildConnectionKey",
    value: function buildConnectionKey(name) {
      return 'connection.' + this.projectId + '.' + v1_default()();
    }
  }, {
    key: "getCurrentConnection",
    value: function getCurrentConnection() {
      return JSON.parse(localStorage.getItem('connection.' + this.projectId + '.current') || '{}');
    }
  }, {
    key: "setCurrentConnection",
    value: function setCurrentConnection(connectionInfo) {
      localStorage.setItem('connection.' + this.projectId + '.current', JSON.stringify(connectionInfo));
    }
  }, {
    key: "getStringCatalog",
    value: function getStringCatalog() {
      return localStorage.getItem('neo4jEtlConnections.' + this.projectId);
    }
  }, {
    key: "setStringCatalog",
    value: function setStringCatalog(catalog) {
      localStorage.setItem('neo4jEtlConnections.' + this.projectId, JSON.stringify(catalog));
    }
  }, {
    key: "stringCatalogToArray",
    value: function stringCatalogToArray() {
      var stringCatalog = this.getStringCatalog();
      var catalog = stringCatalog ? JSON.parse(stringCatalog) : [];
      return catalog;
    }
  }, {
    key: "addConnectionToCatalog",
    value: function addConnectionToCatalog(name) {
      var catalog = this.stringCatalogToArray();
      if (catalog.indexOf(name) === -1) {
        catalog.push(name);
        this.setStringCatalog(catalog);
      }
    }
  }, {
    key: "removeConnectionFromCatalog",
    value: function removeConnectionFromCatalog(name) {
      var catalog = this.stringCatalogToArray();
      // remove the item through identity
      catalog = catalog.filter(function (n) {
        return n !== name;
      });
      this.setStringCatalog(catalog);
    }
  }, {
    key: "saveConnection",
    value: function saveConnection(connectionInfo) {
      if (!connectionInfo.key) {
        connectionInfo = ConnectionStorage_objectSpread(ConnectionStorage_objectSpread({}, connectionInfo), {}, {
          key: this.buildConnectionKey(connectionInfo.name)
        });
      }
      localStorage.setItem(connectionInfo.key, JSON.stringify(connectionInfo));
      this.addConnectionToCatalog(connectionInfo.key);
    }
  }, {
    key: "removeConnection",
    value: function removeConnection(connectionInfo) {
      var key = connectionInfo.key;
      if (!key) {
        key = this.buildConnectionKey(connectionInfo.name);
      }
      localStorage.removeItem(key);
      this.removeConnectionFromCatalog(key);
    }
  }, {
    key: "getConnectionByKey",
    value: function getConnectionByKey(key) {
      var stringConn = localStorage.getItem(key);
      var connection = JSON.parse(stringConn);
      return ConnectionStorage_objectSpread(ConnectionStorage_objectSpread({}, connection), {}, {
        key: key
      });
    }
  }, {
    key: "getConnectionByName",
    value: function getConnectionByName(name) {
      var key = this.buildConnectionKey(name);
      var stringConn = localStorage.getItem(key);
      return JSON.parse(stringConn);
    }

    //FIXME only for retro compatibility
  }, {
    key: "getConnection",
    value: function getConnection(name) {
      return this.getConnectionByName(name);
    }
  }, {
    key: "getAllConnections",
    value: function getAllConnections() {
      var _this = this;
      var stringCatalog = this.getStringCatalog();
      var catalog = stringCatalog ? JSON.parse(stringCatalog) : [];
      return catalog.map(function (key) {
        return _this.getConnectionByKey(key);
      });
    }
  }]);
  return ConnectionStorage;
}();
/* harmony default export */ var services_ConnectionStorage = (ConnectionStorage_ConnectionStorage);
// CONCATENATED MODULE: ./services/GraphStorage.js
function GraphStorage_typeof(obj) { "@babel/helpers - typeof"; return GraphStorage_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, GraphStorage_typeof(obj); }
function _regeneratorRuntime() { "use strict"; /*! regenerator-runtime -- Copyright (c) 2014-present, Facebook, Inc. -- license (MIT): https://github.com/facebook/regenerator/blob/main/LICENSE */ _regeneratorRuntime = function _regeneratorRuntime() { return exports; }; var exports = {}, Op = Object.prototype, hasOwn = Op.hasOwnProperty, defineProperty = Object.defineProperty || function (obj, key, desc) { obj[key] = desc.value; }, $Symbol = "function" == typeof Symbol ? Symbol : {}, iteratorSymbol = $Symbol.iterator || "@@iterator", asyncIteratorSymbol = $Symbol.asyncIterator || "@@asyncIterator", toStringTagSymbol = $Symbol.toStringTag || "@@toStringTag"; function define(obj, key, value) { return Object.defineProperty(obj, key, { value: value, enumerable: !0, configurable: !0, writable: !0 }), obj[key]; } try { define({}, ""); } catch (err) { define = function define(obj, key, value) { return obj[key] = value; }; } function wrap(innerFn, outerFn, self, tryLocsList) { var protoGenerator = outerFn && outerFn.prototype instanceof Generator ? outerFn : Generator, generator = Object.create(protoGenerator.prototype), context = new Context(tryLocsList || []); return defineProperty(generator, "_invoke", { value: makeInvokeMethod(innerFn, self, context) }), generator; } function tryCatch(fn, obj, arg) { try { return { type: "normal", arg: fn.call(obj, arg) }; } catch (err) { return { type: "throw", arg: err }; } } exports.wrap = wrap; var ContinueSentinel = {}; function Generator() {} function GeneratorFunction() {} function GeneratorFunctionPrototype() {} var IteratorPrototype = {}; define(IteratorPrototype, iteratorSymbol, function () { return this; }); var getProto = Object.getPrototypeOf, NativeIteratorPrototype = getProto && getProto(getProto(values([]))); NativeIteratorPrototype && NativeIteratorPrototype !== Op && hasOwn.call(NativeIteratorPrototype, iteratorSymbol) && (IteratorPrototype = NativeIteratorPrototype); var Gp = GeneratorFunctionPrototype.prototype = Generator.prototype = Object.create(IteratorPrototype); function defineIteratorMethods(prototype) { ["next", "throw", "return"].forEach(function (method) { define(prototype, method, function (arg) { return this._invoke(method, arg); }); }); } function AsyncIterator(generator, PromiseImpl) { function invoke(method, arg, resolve, reject) { var record = tryCatch(generator[method], generator, arg); if ("throw" !== record.type) { var result = record.arg, value = result.value; return value && "object" == GraphStorage_typeof(value) && hasOwn.call(value, "__await") ? PromiseImpl.resolve(value.__await).then(function (value) { invoke("next", value, resolve, reject); }, function (err) { invoke("throw", err, resolve, reject); }) : PromiseImpl.resolve(value).then(function (unwrapped) { result.value = unwrapped, resolve(result); }, function (error) { return invoke("throw", error, resolve, reject); }); } reject(record.arg); } var previousPromise; defineProperty(this, "_invoke", { value: function value(method, arg) { function callInvokeWithMethodAndArg() { return new PromiseImpl(function (resolve, reject) { invoke(method, arg, resolve, reject); }); } return previousPromise = previousPromise ? previousPromise.then(callInvokeWithMethodAndArg, callInvokeWithMethodAndArg) : callInvokeWithMethodAndArg(); } }); } function makeInvokeMethod(innerFn, self, context) { var state = "suspendedStart"; return function (method, arg) { if ("executing" === state) throw new Error("Generator is already running"); if ("completed" === state) { if ("throw" === method) throw arg; return doneResult(); } for (context.method = method, context.arg = arg;;) { var delegate = context.delegate; if (delegate) { var delegateResult = maybeInvokeDelegate(delegate, context); if (delegateResult) { if (delegateResult === ContinueSentinel) continue; return delegateResult; } } if ("next" === context.method) context.sent = context._sent = context.arg;else if ("throw" === context.method) { if ("suspendedStart" === state) throw state = "completed", context.arg; context.dispatchException(context.arg); } else "return" === context.method && context.abrupt("return", context.arg); state = "executing"; var record = tryCatch(innerFn, self, context); if ("normal" === record.type) { if (state = context.done ? "completed" : "suspendedYield", record.arg === ContinueSentinel) continue; return { value: record.arg, done: context.done }; } "throw" === record.type && (state = "completed", context.method = "throw", context.arg = record.arg); } }; } function maybeInvokeDelegate(delegate, context) { var method = delegate.iterator[context.method]; if (undefined === method) { if (context.delegate = null, "throw" === context.method) { if (delegate.iterator["return"] && (context.method = "return", context.arg = undefined, maybeInvokeDelegate(delegate, context), "throw" === context.method)) return ContinueSentinel; context.method = "throw", context.arg = new TypeError("The iterator does not provide a 'throw' method"); } return ContinueSentinel; } var record = tryCatch(method, delegate.iterator, context.arg); if ("throw" === record.type) return context.method = "throw", context.arg = record.arg, context.delegate = null, ContinueSentinel; var info = record.arg; return info ? info.done ? (context[delegate.resultName] = info.value, context.next = delegate.nextLoc, "return" !== context.method && (context.method = "next", context.arg = undefined), context.delegate = null, ContinueSentinel) : info : (context.method = "throw", context.arg = new TypeError("iterator result is not an object"), context.delegate = null, ContinueSentinel); } function pushTryEntry(locs) { var entry = { tryLoc: locs[0] }; 1 in locs && (entry.catchLoc = locs[1]), 2 in locs && (entry.finallyLoc = locs[2], entry.afterLoc = locs[3]), this.tryEntries.push(entry); } function resetTryEntry(entry) { var record = entry.completion || {}; record.type = "normal", delete record.arg, entry.completion = record; } function Context(tryLocsList) { this.tryEntries = [{ tryLoc: "root" }], tryLocsList.forEach(pushTryEntry, this), this.reset(!0); } function values(iterable) { if (iterable) { var iteratorMethod = iterable[iteratorSymbol]; if (iteratorMethod) return iteratorMethod.call(iterable); if ("function" == typeof iterable.next) return iterable; if (!isNaN(iterable.length)) { var i = -1, next = function next() { for (; ++i < iterable.length;) { if (hasOwn.call(iterable, i)) return next.value = iterable[i], next.done = !1, next; } return next.value = undefined, next.done = !0, next; }; return next.next = next; } } return { next: doneResult }; } function doneResult() { return { value: undefined, done: !0 }; } return GeneratorFunction.prototype = GeneratorFunctionPrototype, defineProperty(Gp, "constructor", { value: GeneratorFunctionPrototype, configurable: !0 }), defineProperty(GeneratorFunctionPrototype, "constructor", { value: GeneratorFunction, configurable: !0 }), GeneratorFunction.displayName = define(GeneratorFunctionPrototype, toStringTagSymbol, "GeneratorFunction"), exports.isGeneratorFunction = function (genFun) { var ctor = "function" == typeof genFun && genFun.constructor; return !!ctor && (ctor === GeneratorFunction || "GeneratorFunction" === (ctor.displayName || ctor.name)); }, exports.mark = function (genFun) { return Object.setPrototypeOf ? Object.setPrototypeOf(genFun, GeneratorFunctionPrototype) : (genFun.__proto__ = GeneratorFunctionPrototype, define(genFun, toStringTagSymbol, "GeneratorFunction")), genFun.prototype = Object.create(Gp), genFun; }, exports.awrap = function (arg) { return { __await: arg }; }, defineIteratorMethods(AsyncIterator.prototype), define(AsyncIterator.prototype, asyncIteratorSymbol, function () { return this; }), exports.AsyncIterator = AsyncIterator, exports.async = function (innerFn, outerFn, self, tryLocsList, PromiseImpl) { void 0 === PromiseImpl && (PromiseImpl = Promise); var iter = new AsyncIterator(wrap(innerFn, outerFn, self, tryLocsList), PromiseImpl); return exports.isGeneratorFunction(outerFn) ? iter : iter.next().then(function (result) { return result.done ? result.value : iter.next(); }); }, defineIteratorMethods(Gp), define(Gp, toStringTagSymbol, "Generator"), define(Gp, iteratorSymbol, function () { return this; }), define(Gp, "toString", function () { return "[object Generator]"; }), exports.keys = function (val) { var object = Object(val), keys = []; for (var key in object) { keys.push(key); } return keys.reverse(), function next() { for (; keys.length;) { var key = keys.pop(); if (key in object) return next.value = key, next.done = !1, next; } return next.done = !0, next; }; }, exports.values = values, Context.prototype = { constructor: Context, reset: function reset(skipTempReset) { if (this.prev = 0, this.next = 0, this.sent = this._sent = undefined, this.done = !1, this.delegate = null, this.method = "next", this.arg = undefined, this.tryEntries.forEach(resetTryEntry), !skipTempReset) for (var name in this) { "t" === name.charAt(0) && hasOwn.call(this, name) && !isNaN(+name.slice(1)) && (this[name] = undefined); } }, stop: function stop() { this.done = !0; var rootRecord = this.tryEntries[0].completion; if ("throw" === rootRecord.type) throw rootRecord.arg; return this.rval; }, dispatchException: function dispatchException(exception) { if (this.done) throw exception; var context = this; function handle(loc, caught) { return record.type = "throw", record.arg = exception, context.next = loc, caught && (context.method = "next", context.arg = undefined), !!caught; } for (var i = this.tryEntries.length - 1; i >= 0; --i) { var entry = this.tryEntries[i], record = entry.completion; if ("root" === entry.tryLoc) return handle("end"); if (entry.tryLoc <= this.prev) { var hasCatch = hasOwn.call(entry, "catchLoc"), hasFinally = hasOwn.call(entry, "finallyLoc"); if (hasCatch && hasFinally) { if (this.prev < entry.catchLoc) return handle(entry.catchLoc, !0); if (this.prev < entry.finallyLoc) return handle(entry.finallyLoc); } else if (hasCatch) { if (this.prev < entry.catchLoc) return handle(entry.catchLoc, !0); } else { if (!hasFinally) throw new Error("try statement without catch or finally"); if (this.prev < entry.finallyLoc) return handle(entry.finallyLoc); } } } }, abrupt: function abrupt(type, arg) { for (var i = this.tryEntries.length - 1; i >= 0; --i) { var entry = this.tryEntries[i]; if (entry.tryLoc <= this.prev && hasOwn.call(entry, "finallyLoc") && this.prev < entry.finallyLoc) { var finallyEntry = entry; break; } } finallyEntry && ("break" === type || "continue" === type) && finallyEntry.tryLoc <= arg && arg <= finallyEntry.finallyLoc && (finallyEntry = null); var record = finallyEntry ? finallyEntry.completion : {}; return record.type = type, record.arg = arg, finallyEntry ? (this.method = "next", this.next = finallyEntry.finallyLoc, ContinueSentinel) : this.complete(record); }, complete: function complete(record, afterLoc) { if ("throw" === record.type) throw record.arg; return "break" === record.type || "continue" === record.type ? this.next = record.arg : "return" === record.type ? (this.rval = this.arg = record.arg, this.method = "return", this.next = "end") : "normal" === record.type && afterLoc && (this.next = afterLoc), ContinueSentinel; }, finish: function finish(finallyLoc) { for (var i = this.tryEntries.length - 1; i >= 0; --i) { var entry = this.tryEntries[i]; if (entry.finallyLoc === finallyLoc) return this.complete(entry.completion, entry.afterLoc), resetTryEntry(entry), ContinueSentinel; } }, "catch": function _catch(tryLoc) { for (var i = this.tryEntries.length - 1; i >= 0; --i) { var entry = this.tryEntries[i]; if (entry.tryLoc === tryLoc) { var record = entry.completion; if ("throw" === record.type) { var thrown = record.arg; resetTryEntry(entry); } return thrown; } } throw new Error("illegal catch attempt"); }, delegateYield: function delegateYield(iterable, resultName, nextLoc) { return this.delegate = { iterator: values(iterable), resultName: resultName, nextLoc: nextLoc }, "next" === this.method && (this.arg = undefined), ContinueSentinel; } }, exports; }
function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }
function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }
function GraphStorage_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function GraphStorage_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function GraphStorage_createClass(Constructor, protoProps, staticProps) { if (protoProps) GraphStorage_defineProperties(Constructor.prototype, protoProps); if (staticProps) GraphStorage_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
/**
 * manage the connections to graphs
 */
var GraphStorage = /*#__PURE__*/function () {
  function GraphStorage(projectId, context) {
    GraphStorage_classCallCheck(this, GraphStorage);
    this.projectId = projectId || 'default';
    this.context = context;
    this.graphs = this.getGraphsFromContext(context);
  }
  GraphStorage_createClass(GraphStorage, [{
    key: "refreshContext",
    value: function () {
      var _refreshContext = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee() {
        return _regeneratorRuntime().wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return window.neo4jDesktopApi.getContext().then(function (context) {
                  return context;
                });
              case 2:
                return _context.abrupt("return", _context.sent);
              case 3:
              case "end":
                return _context.stop();
            }
          }
        }, _callee);
      }));
      function refreshContext() {
        return _refreshContext.apply(this, arguments);
      }
      return refreshContext;
    }()
  }, {
    key: "getGraphsFromContext",
    value: function getGraphsFromContext(context) {
      var _this = this;
      return context.projects.filter(function (element) {
        return element.id === _this.projectId;
      })[0].graphs;
    }
  }, {
    key: "getCurrentGraph",
    value: function getCurrentGraph() {
      var graphId = localStorage.getItem('graph.' + this.projectId + '.current') || 'default';
      return this.graphs.filter(function (graph) {
        return graph.id === graphId;
      })[0];
    }
  }, {
    key: "setCurrentGraph",
    value: function setCurrentGraph(graph) {
      localStorage.setItem('graph.' + this.projectId + '.current', graph.id);
    }
  }, {
    key: "getAllGraphs",
    value: function getAllGraphs() {
      return this.graphs;
    }
  }]);
  return GraphStorage;
}();
/* harmony default export */ var services_GraphStorage = (GraphStorage);
// CONCATENATED MODULE: ./services/ProjectStorage.js
function ProjectStorage_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ProjectStorage_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ProjectStorage_createClass(Constructor, protoProps, staticProps) { if (protoProps) ProjectStorage_defineProperties(Constructor.prototype, protoProps); if (staticProps) ProjectStorage_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }


var ProjectStorage_ProjectStorage = /*#__PURE__*/function () {
  function ProjectStorage(desktopProject, context) {
    ProjectStorage_classCallCheck(this, ProjectStorage);
    this.original = desktopProject;
    this.connectionStorage = new services_ConnectionStorage(desktopProject.id);
    this.graphStorage = new services_GraphStorage(desktopProject.id, context);
    this.importFolder = context.importFolder;
  }
  ProjectStorage_createClass(ProjectStorage, [{
    key: "addSourceConnection",
    value: function addSourceConnection(sourceConnection) {
      this.connectionStorage.saveConnection(sourceConnection);
    }
  }, {
    key: "removeSourceConnection",
    value: function removeSourceConnection(sourceConnection) {
      this.connectionStorage.removeConnection(sourceConnection);
    }
  }, {
    key: "getAllSourceConnections",
    value: function getAllSourceConnections() {
      return this.connectionStorage.getAllConnections();
    }
  }, {
    key: "getAllGraphs",
    value: function getAllGraphs() {
      return this.graphStorage.getAllGraphs();
    }
  }, {
    key: "getGraphs",
    value: function getGraphs() {
      return this.original.graphs;
    }
  }, {
    key: "getSourceConnectionByName",
    value: function getSourceConnectionByName(name) {
      return this.connectionStorage.getConnectionByName(name);
    }
  }, {
    key: "setCurrentSourceConnection",
    value: function setCurrentSourceConnection(connection) {
      this.connectionStorage.setCurrentConnection(connection);
    }
  }, {
    key: "getCurrentSourceConnection",
    value: function getCurrentSourceConnection() {
      return this.connectionStorage.getCurrentConnection();
    }
  }, {
    key: "getName",
    value: function getName() {
      return this.original.name;
    }
  }, {
    key: "setCurrentGraph",
    value: function setCurrentGraph(graph) {
      this.graphStorage.setCurrentGraph(graph);
    }
  }, {
    key: "getCurrentGraph",
    value: function getCurrentGraph() {
      return this.graphStorage.getCurrentGraph();
    }
  }, {
    key: "getImportFolder",
    value: function getImportFolder() {
      return this.importFolder;
    }
  }]);
  return ProjectStorage;
}();
/* harmony default export */ var services_ProjectStorage = (ProjectStorage_ProjectStorage);
// CONCATENATED MODULE: ./services/ProcessListener.js
function ProcessListener_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ProcessListener_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ProcessListener_createClass(Constructor, protoProps, staticProps) { if (protoProps) ProcessListener_defineProperties(Constructor.prototype, protoProps); if (staticProps) ProcessListener_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
/**
 * Wrapper to manage the java process
 */
var ProcessListener = /*#__PURE__*/function () {
  function ProcessListener() {
    var success = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : function (x) {};
    var fail = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : function (x) {};
    var outListener = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : function (out) {};
    var errListener = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : function (err) {};
    ProcessListener_classCallCheck(this, ProcessListener);
    this.outLog = "";
    this.errLog = "";
    this.success = success;
    this.fail = fail;
    this.outListener = outListener;
    this.errListener = errListener;
    this.context = {};
  }
  ProcessListener_createClass(ProcessListener, [{
    key: "setSuccessHandler",
    value: function setSuccessHandler(success) {
      this.success = success;
    }
  }, {
    key: "setFailureHandler",
    value: function setFailureHandler(fail) {
      this.fail = fail;
    }
  }, {
    key: "setOutputListener",
    value: function setOutputListener(outListener) {
      this.outListener = outListener;
    }
  }, {
    key: "setErrorListener",
    value: function setErrorListener(errListener) {
      this.errListener = errListener;
    }
  }, {
    key: "setContext",
    value: function setContext(context) {
      this.context = context;
    }
  }, {
    key: "getContext",
    value: function getContext() {
      return this.context;
    }

    /**
     * Build the listener for the java process
     */
  }, {
    key: "listener",
    value: function listener() {
      var ref = this;
      return function (process) {
        process.addErrListener(function (err) {
          ref.errLog += err;
          ref.errListener(err);
          console.error(err);
        });
        process.addOutListener(function (out) {
          ref.outLog += out;
          ref.outListener(out);
          console.debug(out);
        });
        process.onExit(function (exitStatus) {
          if (exitStatus === "STOPPED") {
            var result = {
              stdout: ref.outLog,
              stderr: ref.errLog
            };
            ref.success(result);
          }
          if (exitStatus === "KILLED") {
            ref.fail(ref.errLog);
          }
        });
        //process.onError((err)=>{console.error("",err)})
      };
    }
  }]);
  return ProcessListener;
}(); //ProcessListener

/* harmony default export */ var services_ProcessListener = (ProcessListener);
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/lodash/lodash.js
var lodash = __webpack_require__("../../node_modules/lodash/lodash.js");
var lodash_default = /*#__PURE__*/__webpack_require__.n(lodash);

// CONCATENATED MODULE: ../shared/modules/projects/projectDuck.js




var projectDuck_NAME = "projectStore";
var PROJECTS_LOADED = "".concat(projectDuck_NAME, "/LOADED");
var PROJECT_SELECTED = "".concat(projectDuck_NAME, "/SELECTED");
var JAR_NAME = 'dist/neo4j-etl.jar';
var ETL_FILE_UTILS_CLASS = 'org.neo4j.etl.util.FileUtils';
var projectDuck_initialState = {
  projects: [],
  currentProject: {},
  timeRef: 0
};

// Epics
var projectDuck_neo4jDesktopEpic = function neo4jDesktopEpic(action$, store) {
  return action$.ofType(APP_START)["do"](function (action) {
    window.neo4jDesktopApi.getContext().then(function (context) {
      getImportFolderPath().then(function (importFolder) {
        context.importFolder = importFolder;
        var projects = context.projects.map(function (p) {
          return new services_ProjectStorage(p, context);
        });
        store.dispatch({
          type: PROJECTS_LOADED,
          projects: projects
        });
        store.dispatch({
          type: PROJECT_SELECTED,
          currentProject: projects[0]
        });
      });
    });
    window.neo4jDesktopApi.onContextUpdate(function (event, newContext, oldContext) {
      if (!oldContext.importFolder) {
        getImportFolderPath().then(function (importFolder) {
          oldContext.importFolder = importFolder;
          onContextUpdate(newContext, oldContext, store);
        });
      } else {
        onContextUpdate(newContext, oldContext, store);
      }
    });
    return "";
  }).mapTo({
    type: "NOOP"
  });
};
function onContextUpdate(newContext, oldContext, store) {
  newContext.importFolder = oldContext.importFolder;
  var projects = newContext.projects.map(function (p) {
    return new services_ProjectStorage(p, newContext);
  });
  var projectStore = store.getState()['projectStore'];
  var currentProject = lodash_default.a.find(projects, {
    original: {
      id: projectStore.currentProject.original.id
    }
  }, projects[0]);
  store.dispatch({
    type: PROJECTS_LOADED,
    projects: projects
  });
  store.dispatch({
    type: PROJECT_SELECTED,
    currentProject: currentProject
  });
}
function reloadCurrentProject(projects, current) {
  var res = projects[0];
  if (current.original) {
    var find = projects.filter(function (p) {
      return p.original.id === current.original.id;
    })[0];
    if (find) {
      res = find;
    }
  }
  return res;
}
function getImportFolderPath() {
  return new Promise(function (resolve, reject) {
    var processListener = new services_ProcessListener();
    var parameters = {
      options: '',
      classpath: [JAR_NAME],
      "class": ETL_FILE_UTILS_CLASS,
      arguments: ['getimportfolder']
    };
    processListener.setSuccessHandler(function (_ref) {
      var stdout = _ref.stdout,
        stderror = _ref.stderror;
      resolve(stdout);
    });
    processListener.setFailureHandler(function (error) {
      console.error(error);
      reject(error);
    });
    processListener.setErrorListener(function (error) {
      console.error(error);
      reject(error);
    });
    window.neo4jDesktopApi.executeJava(parameters).then(processListener.listener(), processListener.errListener);
  });
}

//reducer
function projectDuck_reducer() {
  var state = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : projectDuck_initialState;
  var action = arguments.length > 1 ? arguments[1] : undefined;
  var timeRef = Date.now();
  switch (action.type) {
    case PROJECTS_LOADED:
      //reload the current graph, if possible
      var currentProject = reloadCurrentProject(action.projects, state.currentProject);
      return Object.assign({}, state, {
        projects: action.projects,
        currentProject: currentProject,
        timeRef: timeRef
      });
    case PROJECT_SELECTED:
      return Object.assign({}, state, {
        currentProject: action.currentProject,
        timeRef: timeRef
      });
    default:
      return state;
  }
}
// CONCATENATED MODULE: ../shared/rootReducer.js
var _stream$settings$comm;
function rootReducer_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */





/* harmony default export */ var rootReducer = (_stream$settings$comm = {}, rootReducer_defineProperty(_stream$settings$comm, streamDuck_NAME, streamDuck_reducer), rootReducer_defineProperty(_stream$settings$comm, NAME, settings), rootReducer_defineProperty(_stream$settings$comm, commandsDuck_NAME, commandsDuck_reducer), rootReducer_defineProperty(_stream$settings$comm, mappingDuck_NAME, mappingDuck_reducer), rootReducer_defineProperty(_stream$settings$comm, projectDuck_NAME, projectDuck_reducer), _stream$settings$comm);
// CONCATENATED MODULE: ../shared/modules/localstorage/localstorageDuck.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var localstorageDuck_NAME = "localstorage";
var CLEAR_LOCALSTORAGE = "".concat(localstorageDuck_NAME, "/CLEAR_LOCALSTORAGE");

// Epics
var clearLocalstorageEpic = function clearLocalstorageEpic(some$, store) {
  return some$.ofType(CLEAR_LOCALSTORAGE).mapTo({
    type: "NOOP"
  });
};
// CONCATENATED MODULE: ../shared/rootEpic.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */





/* harmony default export */ var rootEpic = (Object(es["combineEpics"])(clearLocalstorageEpic, streamDuck_maxFramesConfigEpic, projectDuck_neo4jDesktopEpic));
// CONCATENATED MODULE: ../shared/services/duckUtils.js
function duckUtils_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function duckUtils_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? duckUtils_ownKeys(Object(source), !0).forEach(function (key) { duckUtils_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : duckUtils_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function duckUtils_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var hydrate = function hydrate(initialState, state) {
  if (!state) return state;
  return state.hydrated ? state : duckUtils_objectSpread(duckUtils_objectSpread(duckUtils_objectSpread({}, initialState), state), {}, {
    hydrated: true
  });
};
var dehydrate = function dehydrate(state) {
  if (state) {
    delete state.hydrated;
  }
  return state;
};
// CONCATENATED MODULE: ../shared/services/localstorage.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


var keyPrefix = 'neo4j.';
var storage = window.localStorage;
var localstorage_keys = [];
function getItem(key) {
  try {
    var serializedVal = storage.getItem(keyPrefix + key);
    if (serializedVal === null) return undefined;
    var parsedVal = JSON.parse(serializedVal);
    return parsedVal;
  } catch (e) {
    return undefined;
  }
}
function setItem(key, val) {
  try {
    var serializedVal = JSON.stringify(val);
    storage.setItem(keyPrefix + key, serializedVal);
    return true;
  } catch (e) {
    return false;
  }
}
function getAll() {
  var out = {};
  localstorage_keys.forEach(function (key) {
    var current = getItem(key);
    if (current !== undefined) {
      out[key] = current;
    }
  });
  return out;
}
var localstorage_clear = function clear() {
  return storage.clear();
};
function createReduxMiddleware() {
  return function (store) {
    return function (next) {
      return function (action) {
        var result = next(action);
        var state = store.getState();
        localstorage_keys.forEach(function (key) {
          return setItem(key, dehydrate(state[key]));
        });
        return result;
      };
    };
  };
}
function applyKeys() {
  Array.from(arguments).forEach(function (arg) {
    return localstorage_keys.push(arg);
  });
}
var setPrefix = function setPrefix(p) {
  return keyPrefix = p;
};
var setStorage = function setStorage(s) {
  return storage = s;
};
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/material-design-icons/iconfont/material-icons.css
var material_icons = __webpack_require__("../../node_modules/material-design-icons/iconfont/material-icons.css");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/typeface-roboto/index.css
var typeface_roboto = __webpack_require__("../../node_modules/typeface-roboto/index.css");

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/styles/index.js
var styles = __webpack_require__("../../node_modules/@material-ui/core/styles/index.js");

// CONCATENATED MODULE: ./styles/MUITheme.js

var MUITheme_theme = Object(styles["createMuiTheme"])({
  palette: {
    primary: {
      main: '#0f5788'
    },
    secondary: {
      main: '#f58220'
    }
  },
  typography: {
    fontFamily: '"Open Sans","Helvetica Neue",Helvetica,Roboto,Arial,sans-serif'
  }
});
/* harmony default export */ var MUITheme = (MUITheme_theme);
// CONCATENATED MODULE: ./styles/themes.js
function themes_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function themes_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? themes_ownKeys(Object(source), !0).forEach(function (key) { themes_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : themes_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function themes_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var base = {
  name: 'base',
  // Text colors
  primaryText: '#333',
  secondaryText: '#717172',
  headerText: '#333',
  asideText: '#666',
  link: '#428BCA',
  linkHover: '#5dade2',
  editorCommandColor: '#333',
  topicText: '#428BCA',
  preText: '#333',
  // Backgrounds
  primaryBackground: '#D2D5DA',
  secondaryBackground: '#fff',
  editorBarBackground: '#EFEFF4',
  editorBackground: '#fff',
  drawerBackground: '#30333a',
  topicBackground: '#f8f8f8',
  preBackground: '#f5f5f5',
  alteringTableRowBackground: '#f5f5f5',
  // Fonts
  primaryFontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
  drawerHeaderFontFamily: "'Open Sans', 'HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', Helvetica, Arial, sans-serif",
  streamlineFontFamily: 'streamline',
  editorFont: '"Inconsolata", "Monaco", "Lucida Console", Courier, monospace;',
  // Headers
  primaryHeaderText: '#fff',
  // User feedback colors
  success: '#65B144',
  error: '#E74C3C',
  warning: '#FD952C',
  auth: '#008cc1',
  // Buttons
  primaryButtonText: '#fff',
  primaryButtonBackground: '#008cc1',
  secondaryButtonText: '#717172',
  secondaryButtonBorder: '1px solid #717172',
  secondaryButtonBackground: 'transparent',
  secondaryButtonTextHover: '#fff',
  secondaryButtonBorderHover: '1px solid #717172',
  secondaryButtonBackgroundHover: '#717172',
  formButtonBorder: '1px solid #ccc',
  formButtonBorderHover: '1px solid ##adadad',
  formButtonBackgroundHover: '#e6e6e6',
  editModeButtonText: '#ffaf00',
  // Borders
  frameBorder: 'none',
  inFrameBorder: '1px solid #e6e9ef',
  topicBorder: '1px solid #dadada',
  // Frame
  frameSidebarBackground: '#F8F9FB',
  frameTitlebarText: '#717172'
};
var normal = themes_objectSpread(themes_objectSpread({}, base), {}, {
  name: 'normal'
});
var outline = themes_objectSpread(themes_objectSpread({}, base), {}, {
  name: 'outline',
  primaryText: '#000',
  secondaryText: '#000',
  frameBorder: '1px solid #000',
  inFrameBorder: '1px solid #000',
  topicBorder: '1px solid #000'
});
var dark = themes_objectSpread(themes_objectSpread({}, base), {}, {
  name: 'dark',
  primaryText: '#f4f4f4',
  secondaryText: '#eee',
  headerText: '#f4f4f4',
  primaryHeaderText: '#f4f4f4',
  editorCommandColor: '#1abc9c',
  link: '#f4f4f4',
  topicText: '#fff',
  preText: '#fff',
  // Backgrounds
  primaryBackground: '#282c32',
  secondaryBackground: '#5a6070',
  editorBarBackground: '#5a6070',
  editorBackground: '#282c32',
  drawerBackground: '#30333a',
  frameSidebarBackground: '#5a6070',
  topicBackground: 'transparent',
  preBackground: '#282c32',
  alteringTableRowBackground: '#282c32',
  // Buttons
  primaryButtonText: '#fff',
  primaryButtonBackground: '#008cc1',
  secondaryButtonText: '#f4f4f4',
  secondaryButtonBorder: '1px solid #717172',
  secondaryButtonBackground: 'transparent',
  secondaryButtonTextHover: '#f4f4f4',
  secondaryButtonBorderHover: '1px solid #717172',
  secondaryButtonBackgroundHover: '#282c32',
  // Borders
  inFrameBorder: '1px solid #f4f4f4'
});
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/CssBaseline/index.js
var CssBaseline = __webpack_require__("../../node_modules/@material-ui/core/CssBaseline/index.js");
var CssBaseline_default = /*#__PURE__*/__webpack_require__.n(CssBaseline);

// CONCATENATED MODULE: ./modules/App/styled.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


var StyledWrapper = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledWrapper",
  componentId: "sc-e4i7ib-0"
})(["display:flex;flex-direction:row;height:100vh;font-family:", ";font-size:13px;-webkit-font-smoothing:antialiased;overflow:hidden;"], function (props) {
  return props.theme.primaryFontFamily;
});
var StyledApp = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledApp",
  componentId: "sc-e4i7ib-1"
})(["flex-grow:1;flex-shrink:1;flex-basis:0%;display:flex;flex-direction:column;height:100vh;width:100vw;"]);
var StyledBody = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledBody",
  componentId: "sc-e4i7ib-2"
})(["flex:auto;display:flex;flex-direction:row;height:inherit;"]);
var StyledMainWrapper = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledMainWrapper",
  componentId: "sc-e4i7ib-3"
})(["flex:auto;overflow:auto;padding:0;z-index:1;height:auto;width:0;background-color:", ";color:", ";"], function (props) {
  return props.theme.primaryBackground;
}, function (props) {
  return props.theme.primaryText;
});
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Paper/index.js
var Paper = __webpack_require__("../../node_modules/@material-ui/core/Paper/index.js");
var Paper_default = /*#__PURE__*/__webpack_require__.n(Paper);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Grid/index.js
var Grid = __webpack_require__("../../node_modules/@material-ui/core/Grid/index.js");
var Grid_default = /*#__PURE__*/__webpack_require__.n(Grid);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Select/index.js
var Select = __webpack_require__("../../node_modules/@material-ui/core/Select/index.js");
var Select_default = /*#__PURE__*/__webpack_require__.n(Select);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Icon/index.js
var Icon = __webpack_require__("../../node_modules/@material-ui/core/Icon/index.js");
var Icon_default = /*#__PURE__*/__webpack_require__.n(Icon);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Chip/index.js
var Chip = __webpack_require__("../../node_modules/@material-ui/core/Chip/index.js");
var Chip_default = /*#__PURE__*/__webpack_require__.n(Chip);

// CONCATENATED MODULE: ./styles/constants.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var font = {};
var dim = {
  // Editor bar
  editorbarHeight: 70,
  // Frame
  frameBodyHeight: 550,
  frameTitlebarHeight: 39,
  frameStatusbarHeight: 39,
  frameBodyPadding: 20
};
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/classnames/index.js
var classnames = __webpack_require__("../../node_modules/classnames/index.js");
var classnames_default = /*#__PURE__*/__webpack_require__.n(classnames);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Card/index.js
var Card = __webpack_require__("../../node_modules/@material-ui/core/Card/index.js");
var Card_default = /*#__PURE__*/__webpack_require__.n(Card);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/CardContent/index.js
var CardContent = __webpack_require__("../../node_modules/@material-ui/core/CardContent/index.js");
var CardContent_default = /*#__PURE__*/__webpack_require__.n(CardContent);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Radio/index.js
var Radio = __webpack_require__("../../node_modules/@material-ui/core/Radio/index.js");
var Radio_default = /*#__PURE__*/__webpack_require__.n(Radio);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Badge/index.js
var Badge = __webpack_require__("../../node_modules/@material-ui/core/Badge/index.js");
var Badge_default = /*#__PURE__*/__webpack_require__.n(Badge);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Dialog/index.js
var Dialog = __webpack_require__("../../node_modules/@material-ui/core/Dialog/index.js");
var Dialog_default = /*#__PURE__*/__webpack_require__.n(Dialog);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/DialogTitle/index.js
var DialogTitle = __webpack_require__("../../node_modules/@material-ui/core/DialogTitle/index.js");
var DialogTitle_default = /*#__PURE__*/__webpack_require__.n(DialogTitle);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/DialogActions/index.js
var DialogActions = __webpack_require__("../../node_modules/@material-ui/core/DialogActions/index.js");
var DialogActions_default = /*#__PURE__*/__webpack_require__.n(DialogActions);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Button/index.js
var Button = __webpack_require__("../../node_modules/@material-ui/core/Button/index.js");
var Button_default = /*#__PURE__*/__webpack_require__.n(Button);

// CONCATENATED MODULE: ./modules/Stream/GraphChip.jsx
function GraphChip_typeof(obj) { "@babel/helpers - typeof"; return GraphChip_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, GraphChip_typeof(obj); }
var _excluded = ["classes", "children", "isSelected", "icon", "onDelete", "dataId", "allowEdit", "disabled"];
function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
function _objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = _objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }
function _objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }
function GraphChip_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function GraphChip_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function GraphChip_createClass(Constructor, protoProps, staticProps) { if (protoProps) GraphChip_defineProperties(Constructor.prototype, protoProps); if (staticProps) GraphChip_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) _setPrototypeOf(subClass, superClass); }
function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }
function _createSuper(Derived) { var hasNativeReflectConstruct = _isNativeReflectConstruct(); return function _createSuperInternal() { var Super = _getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = _getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return _possibleConstructorReturn(this, result); }; }
function _possibleConstructorReturn(self, call) { if (call && (GraphChip_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return _assertThisInitialized(self); }
function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }







var GraphChip_GraphChip = /*#__PURE__*/function (_React$Component) {
  _inherits(GraphChip, _React$Component);
  var _super = _createSuper(GraphChip);
  function GraphChip(props) {
    var _this;
    GraphChip_classCallCheck(this, GraphChip);
    _this = _super.call(this, props);
    _this.state = {
      open: false
    };
    _this.handleOpen = _this.handleOpen.bind(_assertThisInitialized(_this));
    _this.handleClose = _this.handleClose.bind(_assertThisInitialized(_this));
    _this.startEdit = _this.startEdit.bind(_assertThisInitialized(_this));
    return _this;
  }
  GraphChip_createClass(GraphChip, [{
    key: "handleOpen",
    value: function handleOpen() {
      this.setState({
        open: true
      });
    }
  }, {
    key: "handleClose",
    value: function handleClose() {
      this.setState({
        open: false
      });
    }
  }, {
    key: "handleDelete",
    value: function handleDelete(onDelete) {
      onDelete();
      this.handleClose();
    }
  }, {
    key: "startEdit",
    value: function startEdit(e) {
      e.preventDefault();
      // Remove current frame
      this.props.bus.send(REMOVE, {
        id: 'MAPPING'
      });

      // Add connection frame
      this.props.bus.send(ADD, {
        state: {
          id: 'RDBMS',
          type: 'connection',
          connection: this.props.connection
        }
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this$props = this.props,
        classes = _this$props.classes,
        children = _this$props.children,
        isSelected = _this$props.isSelected,
        icon = _this$props.icon,
        onDelete = _this$props.onDelete,
        dataId = _this$props.dataId,
        allowEdit = _this$props.allowEdit,
        disabled = _this$props.disabled,
        others = _objectWithoutProperties(_this$props, _excluded);
      var hasDelete = typeof onDelete === 'function';
      var card = /*#__PURE__*/react_default.a.createElement(Card_default.a, _extends({
        classes: {
          root: classnames_default()(classes.cardRoot, !hasDelete && classes.cardMargin, disabled && classes.cardDisabled, isSelected && (disabled ? classes.cardSelectedDisabled : classes.cardSelected))
        },
        "data-id": !hasDelete && dataId
      }, others), /*#__PURE__*/react_default.a.createElement(CardContent_default.a, {
        classes: {
          root: classes.cardContent
        }
      }, /*#__PURE__*/react_default.a.createElement("div", {
        className: classes.icon
      }, icon, /*#__PURE__*/react_default.a.createElement(Radio_default.a, {
        checked: isSelected,
        color: "primary",
        classes: {
          root: classes.radioRoot
        },
        disabled: disabled
      })), /*#__PURE__*/react_default.a.createElement("div", {
        className: classes.text
      }, children), allowEdit && /*#__PURE__*/react_default.a.createElement("span", {
        className: classes.editButton
      }, /*#__PURE__*/react_default.a.createElement(Icon_default.a, {
        onClick: this.startEdit,
        classes: {
          root: classes.editButtonIcon
        },
        disabled: disabled
      }, "edit"))));
      if (hasDelete) {
        return /*#__PURE__*/react_default.a.createElement(react_default.a.Fragment, null, /*#__PURE__*/react_default.a.createElement(Badge_default.a, {
          badgeContent: /*#__PURE__*/react_default.a.createElement(Icon_default.a, {
            onClick: this.handleOpen,
            disabled: disabled
          }, "close"),
          color: "primary",
          classes: {
            root: classes.cardMargin,
            badge: classes.deleteIcon
          },
          "data-id": dataId
        }, card), /*#__PURE__*/react_default.a.createElement(Dialog_default.a, {
          open: this.state.open,
          onClose: this.handleClose
        }, /*#__PURE__*/react_default.a.createElement(DialogTitle_default.a, null, "Are you sure you want to delete this connection?"), /*#__PURE__*/react_default.a.createElement(DialogActions_default.a, null, /*#__PURE__*/react_default.a.createElement(Button_default.a, {
          onClick: this.handleClose,
          color: "primary"
        }, "Cancel"), /*#__PURE__*/react_default.a.createElement(Button_default.a, {
          onClick: this.handleDelete.bind(this, onDelete),
          color: "primary",
          autoFocus: true
        }, "Delete"))));
      } else {
        return card;
      }
    }
  }]);
  return GraphChip;
}(react_default.a.Component);
var StyledGraphChip = Object(styles["withStyles"])(function (theme) {
  return {
    cardRoot: {
      cursor: 'pointer',
      display: 'inline-block',
      backgroundColor: theme.palette.grey['300'],
      '&:hover': {
        backgroundColor: theme.palette.grey['400']
      }
    },
    cardDisabled: {
      cursor: 'default',
      backgroundColor: theme.palette.grey['200'],
      '&:hover': {
        backgroundColor: theme.palette.grey['200']
      }
    },
    cardMargin: {
      margin: "0 ".concat(theme.spacing.unit * 2, "px ").concat(theme.spacing.unit * 2, "px 0")
    },
    cardSelected: {
      backgroundColor: theme.palette.grey['500']
    },
    cardSelectedDisabled: {
      cursor: 'default',
      backgroundColor: theme.palette.grey['400'],
      '&:hover': {
        backgroundColor: theme.palette.grey['400']
      }
    },
    cardContent: {
      display: 'flex'
    },
    icon: {
      width: theme.spacing.unit * 5,
      marginLeft: theme.spacing.unit * -1
    },
    editButton: {
      position: 'absolute',
      top: '-11px',
      right: '18px',
      background: theme.palette.grey['A100'],
      borderRadius: '50%',
      padding: '2px 3px 0 3px',
      cursor: 'pointer'
    },
    editButtonIcon: {
      fontSize: '18px'
    },
    text: {
      display: 'flex',
      flexDirection: 'column'
    },
    deleteIcon: {
      cursor: 'pointer',
      color: '#fff',
      backgroundColor: theme.palette.grey['A200'],
      fontSize: '18px'
    },
    radioRoot: {
      padding: "0 ".concat(theme.spacing.unit, "px")
    }
  };
})(Object(react_suber_lib["withBus"])(GraphChip_GraphChip));
// CONCATENATED MODULE: ./modules/Stream/styled.jsx
var styled_excluded = ["classes"],
  _excluded2 = ["classes"],
  _excluded3 = ["classes"],
  _excluded4 = ["innerRef"],
  _excluded5 = ["classes"],
  _excluded6 = ["classes"],
  _excluded7 = ["classes"],
  _excluded8 = ["isSelected", "classes"],
  _excluded9 = ["selected"],
  _excluded10 = ["classes"],
  _excluded11 = ["classes", "color"];
function styled_extends() { styled_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return styled_extends.apply(this, arguments); }
function styled_objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = styled_objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }
function styled_objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }
function styled_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */







var StyledStream = Object(styles["withStyles"])({
  root: styled_defineProperty({
    padding: 0,
    display: 'flex',
    flexDirection: 'column',
    marginTop: '17px',
    overflow: 'auto'
  }, "padding", '0px 24px')
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, styled_excluded);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});

// Frames
var StyledFrame = Object(styles["withStyles"])(function (theme) {
  return {
    '@keyframes rollDownAnimation': {
      from: {
        transform: "translate(0, -".concat(dim.frameBodyHeight, "px)"),
        maxHeight: 0
      },
      to: {
        transform: 'translateY(0)',
        maxHeight: '500px' /* Greater than a frame can be */
      }
    },

    root: {
      animation: 'rollDownAnimation 0.4s ease-in',
      padding: theme.spacing.unit * 2,
      marginBottom: theme.spacing.unit * 2
    }
  };
})(function (props) {
  return /*#__PURE__*/react_default.a.createElement(Paper_default.a, props);
});
var styled_StyledFrameSlide = function StyledFrameSlide(props) {
  return /*#__PURE__*/react_default.a.createElement(Grid_default.a, styled_extends({
    container: true,
    spacing: 24,
    component: "article"
  }, props));
};
var StyledFrameBody = Object(styles["withStyles"])({
  root: {
    display: 'flex',
    flexDirection: 'row',
    width: '100%'
  }
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded2);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});
var StyledFrameMainSection = Object(styles["withStyles"])({
  root: {
    minWidth: 0,
    flex: '1 1 auto',
    height: 'inherit',
    display: 'flex',
    flexDirection: 'column',
    width: '100%'
  }
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded3);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});
var styled_StyledFrameContents = function StyledFrameContents(props) {
  var innerRef = props.innerRef,
    other = styled_objectWithoutProperties(props, _excluded4);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    ref: innerRef
  }, other));
};
var styled_StyledFrameStatusbar = function StyledFrameStatusbar(props) {
  return /*#__PURE__*/react_default.a.createElement("div", props);
};
var PaddedDiv = Object(styles["withStyles"])({
  root: {
    padding: '0 20px 20px 20px',
    paddingBottom: '20px'
  }
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded5);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});
var StyledFrameTitleBar = Object(styles["withStyles"])(function (theme) {
  return {
    root: {
      borderBottom: "1px solid ".concat(theme.palette.divider)
    }
  };
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded6);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});
var ErrorText = Object(styles["withStyles"])(function (theme) {
  return {
    root: {
      paddingLeft: '5px',
      lineHeight: '35px',
      color: theme.palette.error.main
    }
  };
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded7);
  return /*#__PURE__*/react_default.a.createElement("span", styled_extends({
    className: classes.root
  }, others));
});
var StyledSelect = Select_default.a;
var StyledChipIcon = Object(styles["withStyles"])(function (theme) {
  return {
    root: {
      color: '#717172',
      marginLeft: theme.spacing.unit
    }
  };
})(Icon_default.a);
var StyledConnectionChip = Object(styles["withStyles"])(function (theme) {
  return {
    root: {
      backgroundColor: theme.palette.grey['300'],
      margin: "0 ".concat(theme.spacing.unit * 2, "px ").concat(theme.spacing.unit * 2, "px 0"),
      '&:hover': {
        backgroundColor: theme.palette.grey['400']
      }
    },
    selected: {
      backgroundColor: theme.palette.grey['500']
    }
  };
})(function (props) {
  var isSelected = props.isSelected,
    classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded8);
  var selected = classes.selected,
    otherClasses = styled_objectWithoutProperties(classes, _excluded9);
  return /*#__PURE__*/react_default.a.createElement(Chip_default.a, styled_extends({
    className: isSelected ? selected : null,
    classes: otherClasses
  }, others));
});
var GraphNotAvailable = Object(styles["withStyles"])(function (theme) {
  return {
    root: {
      background: theme.palette.error.light,
      color: theme.palette.error.contrastText,
      padding: theme.spacing.unit,
      marginTop: theme.spacing.unit,
      marginBottom: theme.spacing.unit
    }
  };
})(function (props) {
  var classes = props.classes,
    others = styled_objectWithoutProperties(props, _excluded10);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others));
});
var DotColor = Object(styles["withStyles"])({
  root: {
    width: '20px',
    height: '20px',
    display: 'inline-block',
    border: '1px solid white',
    borderRadius: '50%'
  }
})(function (props) {
  var classes = props.classes,
    color = props.color,
    others = styled_objectWithoutProperties(props, _excluded11);
  return /*#__PURE__*/react_default.a.createElement("div", styled_extends({
    className: classes.root
  }, others, {
    style: {
      backgroundColor: color.background,
      borderColor: color.border
    }
  }));
});
// CONCATENATED MODULE: ./components/Render/index.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var Render = function Render(_ref) {
  var cond = _ref["if"],
    children = _ref.children;
  return cond ? children : null;
};
/* harmony default export */ var components_Render = (Render);
// CONCATENATED MODULE: ./modules/Stream/FrameTemplate.jsx
function FrameTemplate_typeof(obj) { "@babel/helpers - typeof"; return FrameTemplate_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, FrameTemplate_typeof(obj); }
function FrameTemplate_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function FrameTemplate_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function FrameTemplate_createClass(Constructor, protoProps, staticProps) { if (protoProps) FrameTemplate_defineProperties(Constructor.prototype, protoProps); if (staticProps) FrameTemplate_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function FrameTemplate_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) FrameTemplate_setPrototypeOf(subClass, superClass); }
function FrameTemplate_setPrototypeOf(o, p) { FrameTemplate_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return FrameTemplate_setPrototypeOf(o, p); }
function FrameTemplate_createSuper(Derived) { var hasNativeReflectConstruct = FrameTemplate_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = FrameTemplate_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = FrameTemplate_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return FrameTemplate_possibleConstructorReturn(this, result); }; }
function FrameTemplate_possibleConstructorReturn(self, call) { if (call && (FrameTemplate_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return FrameTemplate_assertThisInitialized(self); }
function FrameTemplate_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function FrameTemplate_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function FrameTemplate_getPrototypeOf(o) { FrameTemplate_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return FrameTemplate_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var FrameTemplate_FrameTemplate = /*#__PURE__*/function (_Component) {
  FrameTemplate_inherits(FrameTemplate, _Component);
  var _super = FrameTemplate_createSuper(FrameTemplate);
  function FrameTemplate(props) {
    var _this;
    FrameTemplate_classCallCheck(this, FrameTemplate);
    _this = _super.call(this, props);
    _this.state = {
      lastHeight: 10
    };
    return _this;
  }
  FrameTemplate_createClass(FrameTemplate, [{
    key: "componentDidUpdate",
    value: function componentDidUpdate() {
      if (this.frameContentElement.clientHeight < 300) return; // No need to report a transition
      if (this.frameContentElement && this.state.lastHeight !== this.frameContentElement.clientHeight) {
        this.props.onResize && this.props.onResize(this.frameContentElement.clientHeight);
        this.setState({
          lastHeight: this.frameContentElement.clientHeight
        });
      }
    }
  }, {
    key: "setFrameContentElement",
    value: function setFrameContentElement(el) {
      this.frameContentElement = el;
    }
  }, {
    key: "render",
    value: function render() {
      return /*#__PURE__*/react_default.a.createElement(StyledFrame, null, /*#__PURE__*/react_default.a.createElement(StyledFrameBody, null, this.props.sidebar ? this.props.sidebar() : null, /*#__PURE__*/react_default.a.createElement(StyledFrameMainSection, null, /*#__PURE__*/react_default.a.createElement(styled_StyledFrameContents, {
        innerRef: this.setFrameContentElement.bind(this)
      }, this.props.contents), /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.props.statusbar
      }, /*#__PURE__*/react_default.a.createElement(styled_StyledFrameStatusbar, null, this.props.statusbar)))));
    }
  }]);
  return FrameTemplate;
}(react["Component"]);
/* harmony default export */ var Stream_FrameTemplate = (FrameTemplate_FrameTemplate);
// CONCATENATED MODULE: ./modules/Stream/Frame.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



var Frame_Frame = function Frame(_ref) {
  var frame = _ref.frame;
  var errors = frame.errors || false;
  var contents = frame.contents || false;
  var frameContents = contents;
  if (errors) {
    frameContents = /*#__PURE__*/react_default.a.createElement("div", null, /*#__PURE__*/react_default.a.createElement("pre", null, errors.message));
  } else if (frame.type === 'unknown') {
    frameContents = 'Unknown command';
  }
  return /*#__PURE__*/react_default.a.createElement(Stream_FrameTemplate, {
    header: frame,
    contents: frameContents
  });
};
/* harmony default export */ var Stream_Frame = (Frame_Frame);
// CONCATENATED MODULE: ./modules/Stream/PreFrame.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var PreFrame_PreFrame = function PreFrame(_ref) {
  var frame = _ref.frame;
  return /*#__PURE__*/react_default.a.createElement(Stream_FrameTemplate, {
    header: frame,
    contents: /*#__PURE__*/react_default.a.createElement(PaddedDiv, null, /*#__PURE__*/react_default.a.createElement("pre", null, frame.result || frame.contents))
  });
};
/* harmony default export */ var Stream_PreFrame = (PreFrame_PreFrame);
// CONCATENATED MODULE: ../shared/services/exceptionMessages.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var AddServerValidationError = 'Wrong format. It should be ":server add name username:password@host:port"';
var CreateDataSourceValidationError = 'Wrong format. It should be ":datasource create {"name": "myName", "command": "RETURN rand()", "bookmarkId":"uuid-of-existing-bookmark", "refreshInterval": 10, "parameters": {}}"';
var RemoveDataSourceValidationError = 'Wrong format. It should be ":datasource remove uuid-of-existing-datasource"';
var BoltConnectionError = 'No connection found, did you connect to Neo4j?';
var BoltError = '#code# - #message#';
var Neo4jError = '#message#';
var UnknownCommandError = 'Unknown command #cmd#';
var BookmarkNotFoundError = 'No connection with the name #name# found. Add a bookmark before trying to connect.';
var OpenConnectionNotFoundError = 'No open connection with the name #name# found. You have to connect to a bookmark before you can use it.';
var CouldNotFetchRemoteGuideError = 'Can not fetch remote guide: #error#';
var FetchURLError = 'Could not fetch URL: "#error#". This could be due to the remote server policy. See your web browsers error console for more information.';
// CONCATENATED MODULE: ../shared/services/exceptions.js
function _construct(Parent, args, Class) { if (exceptions_isNativeReflectConstruct()) { _construct = Reflect.construct.bind(); } else { _construct = function _construct(Parent, args, Class) { var a = [null]; a.push.apply(a, args); var Constructor = Function.bind.apply(Parent, a); var instance = new Constructor(); if (Class) exceptions_setPrototypeOf(instance, Class.prototype); return instance; }; } return _construct.apply(null, arguments); }
function exceptions_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function exceptions_setPrototypeOf(o, p) { exceptions_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return exceptions_setPrototypeOf(o, p); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


function exceptions_getErrorMessage(errorObject) {
  var str = exceptionMessages_namespaceObject[errorObject.type];
  if (!str) return;
  var keys = Object.keys(errorObject);
  keys.forEach(function (prop) {
    var re = new RegExp('(#' + prop + '#)', 'g');
    str = str.replace(re, errorObject[prop]);
  });
  return str;
}
function createErrorObject(ErrorType) {
  for (var _len = arguments.length, rest = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    rest[_key - 1] = arguments[_key];
  }
  var obj = _construct(ErrorType, rest);
  if (!obj.code) obj.code = obj.type;
  obj.message = exceptions_getErrorMessage(obj);
  return obj;
}
function UserException(message) {
  return {
    type: 'UserException',
    message: message
  };
}
function ConnectionException(message) {
  var code = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'Connection Error';
  return {
    fields: [{
      code: code,
      message: message
    }]
  };
}
function exceptions_AddServerValidationError() {
  return {
    type: 'AddServerValidationError'
  };
}
function exceptions_CreateDataSourceValidationError() {
  return {
    type: 'CreateDataSourceValidationError'
  };
}
function exceptions_RemoveDataSourceValidationError() {
  return {
    type: 'RemoveDataSourceValidationError'
  };
}
function exceptions_BoltConnectionError() {
  return {
    type: 'BoltConnectionError'
  };
}
function exceptions_BoltError(obj) {
  return {
    type: 'BoltError',
    code: obj.fields[0].code,
    message: obj.fields[0].message
  };
}
function exceptions_Neo4jError(obj) {
  return {
    type: 'Neo4jError',
    message: obj.message
  };
}
function ConnectionNotFoundError(name) {
  return {
    type: 'ConnectionNotFoundError',
    name: name
  };
}
function exceptions_OpenConnectionNotFoundError(name) {
  return {
    type: 'OpenConnectionNotFoundError',
    name: name
  };
}
function exceptions_UnknownCommandError(cmd) {
  return {
    type: 'UnknownCommandError',
    cmd: cmd
  };
}
function exceptions_CouldNotFetchRemoteGuideError(error) {
  return {
    type: 'CouldNotFetchRemoteGuideError',
    error: error
  };
}
function exceptions_FetchURLError(error) {
  return {
    type: 'FetchURLError',
    error: error
  };
}
// CONCATENATED MODULE: ./modules/Stream/ErrorFrame.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */






var ErrorFrame_ErrorFrame = function ErrorFrame(_ref) {
  var frame = _ref.frame;
  var error = frame.error || false;
  var errorContents = error.message || 'No error message found';
  if (error.type && typeof exceptionMessages_namespaceObject[error.type] !== 'undefined') {
    errorContents = exceptions_getErrorMessage(error);
  }
  return /*#__PURE__*/react_default.a.createElement(Stream_FrameTemplate, {
    header: frame,
    contents: /*#__PURE__*/react_default.a.createElement(PaddedDiv, null, /*#__PURE__*/react_default.a.createElement("pre", null, errorContents))
  });
};
/* harmony default export */ var Stream_ErrorFrame = (ErrorFrame_ErrorFrame);
// EXTERNAL MODULE: ./modules/Guides/style.css
var style = __webpack_require__("./modules/Guides/style.css");
var style_default = /*#__PURE__*/__webpack_require__.n(style);

// CONCATENATED MODULE: ./modules/Guides/styled.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


var SlideContainer = styled_components_es["default"].div.withConfig({
  displayName: "styled__SlideContainer",
  componentId: "sc-12e243a-0"
})(["padding:0 60px;width:100%;display:inline-block;"]);
var StyledCarouselLeft = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledCarouselLeft",
  componentId: "sc-12e243a-1"
})(["float:left;"]);
var StyledCarouselRight = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledCarouselRight",
  componentId: "sc-12e243a-2"
})(["float:right;"]);
var StyledCarouselButtonContainer = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledCarouselButtonContainer",
  componentId: "sc-12e243a-3"
})([""]);
var CarouselIndicator = styled_components_es["default"].li.withConfig({
  displayName: "styled__CarouselIndicator",
  componentId: "sc-12e243a-4"
})(["display:inline-block;width:10px;height:10px;margin:1px;text-indent:-999px;cursor:pointer;background-color:transparent;border:1px solid #fff;border-radius:10px;cursor:pointer;background-color:rgba(188,195,207,0.64);"]);
var CarouselIndicatorInactive = Object(styled_components_es["default"])(CarouselIndicator).withConfig({
  displayName: "styled__CarouselIndicatorInactive",
  componentId: "sc-12e243a-5"
})(["&:hover{background-color:rgba(188,195,207,1);}"]);
var CarouselIndicatorActive = Object(styled_components_es["default"])(CarouselIndicator).withConfig({
  displayName: "styled__CarouselIndicatorActive",
  componentId: "sc-12e243a-6"
})(["width:12px;height:12px;background-color:#428bca;"]);
var StyledUl = styled_components_es["default"].ul.withConfig({
  displayName: "styled__StyledUl",
  componentId: "sc-12e243a-7"
})(["margin:15px;"]);
var StyledSlide = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledSlide",
  componentId: "sc-12e243a-8"
})(["color:", ";& p.lead,.title,.subtitle,.content > p,.table-help{color:", " !important;}& a{color:", ";text-decoration:", ";}& kbd{color:", " !important;background-color:", " !important;}& .content > pre{background-color:", ";color:", ";}& pre.runnable{background-color:", ";color:", ";}& pre.content{background-color:", ";color:", ";}& a[help-topic],a[play-topic],a[server-topic],a[exec-topic]{background-color:", " !important;color:", " !important;}& button [help-topic],button [play-topic],button [server-topic],button [exec-topic]{background-color:", ";color:", ";}.code{background-color:transparent;}"], function (props) {
  return props.theme.primaryText;
}, function (props) {
  return props.theme.primaryText;
}, function (props) {
  return props.theme.link;
}, function (props) {
  return props.theme.name === 'dark' ? 'underline' : 'none';
}, function (props) {
  return props.theme.primaryBackground;
}, function (props) {
  return props.theme.primaryText;
}, function (props) {
  return props.theme.secondaryBackground;
}, function (props) {
  return props.theme.preText;
}, function (props) {
  return props.theme.preBackground;
}, function (props) {
  return props.theme.preText;
}, function (props) {
  return props.theme.secondaryBackground;
}, function (props) {
  return props.theme.preText;
}, function (props) {
  return props.theme.topicBackground;
}, function (props) {
  return props.theme.topicText;
}, function (props) {
  return props.theme.primaryButtonBackground;
}, function (props) {
  return props.theme.primaryButtonText;
});
// CONCATENATED MODULE: ./modules/Guides/Slide.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var Slide_Slide = function Slide(_ref) {
  var html = _ref.html;
  return /*#__PURE__*/react_default.a.createElement(StyledSlide, {
    className: style_default.a.slide,
    dangerouslySetInnerHTML: {
      __html: html
    }
  });
};
/* harmony default export */ var Guides_Slide = (Slide_Slide);
// EXTERNAL MODULE: ./modules/Help/html/intro.html
var intro = __webpack_require__("./modules/Help/html/intro.html");
var intro_default = /*#__PURE__*/__webpack_require__.n(intro);

// EXTERNAL MODULE: ./modules/Help/html/help.html
var html_help = __webpack_require__("./modules/Help/html/help.html");
var help_default = /*#__PURE__*/__webpack_require__.n(html_help);

// EXTERNAL MODULE: ./modules/Help/html/unknown.html
var unknown = __webpack_require__("./modules/Help/html/unknown.html");
var unknown_default = /*#__PURE__*/__webpack_require__.n(unknown);

// CONCATENATED MODULE: ./modules/Help/html/index.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




/* harmony default export */ var Help_html = ({
  _intro: intro_default.a,
  _help: help_default.a,
  _unknown: unknown_default.a
});
// CONCATENATED MODULE: ../shared/services/commandUtils.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

function cleanCommand(cmd) {
  var noComments = stripCommandComments(cmd);
  var noEmptyLines = stripEmptyCommandLines(noComments);
  return noEmptyLines;
}
function stripEmptyCommandLines(str) {
  var skipEmptyLines = function skipEmptyLines(e) {
    return !/^\s*$/.test(e);
  };
  return str.split('\n').filter(skipEmptyLines).join('\n');
}
function stripCommandComments(str) {
  return str.replace(/((^|\n)\/\/[^\n$]+\n?)/g, '');
}
function splitStringOnFirst(str, delimiter) {
  var parts = str.split(delimiter);
  return [].concat(parts[0], parts.slice(1).join(delimiter));
}
function splitStringOnLast(str, delimiter) {
  var parts = str.split(delimiter);
  return [].concat(parts.slice(0, parts.length - 1).join(delimiter), parts[parts.length - 1]);
}
var isCypherCommand = function isCypherCommand(cmd, cmdchar) {
  var cleanCmd = cleanCommand(cmd);
  return cleanCmd[0] !== cmdchar;
};
var getInterpreter = function getInterpreter(interpret, cmd, cmdchar) {
  if (isCypherCommand(cmd, cmdchar)) return interpret('cypher');
  return interpret(cleanCommand(cmd).substr(cmdchar.length));
};
var isNamedInterpreter = function isNamedInterpreter(interpreter) {
  return interpreter && interpreter.name !== 'catch-all';
};
var extractPostConnectCommandsFromServerConfig = function extractPostConnectCommandsFromServerConfig(str) {
  var substituteStr = '@@semicolon@@';
  var substituteRe = new RegExp(substituteStr, 'g');
  var replaceFn = function replaceFn(m) {
    return m.replace(/;/g, substituteStr);
  };
  var qs = [/(`[^`]*?`)/g, /("[^"]*?")/g, /('[^']*?')/g];
  qs.forEach(function (q) {
    return str = str.replace(q, replaceFn);
  });
  var splitted = str.split(';').map(function (item) {
    return item.trim();
  }).map(function (item) {
    return item.replace(substituteRe, ';');
  }).filter(function (item) {
    return item && item.length;
  });
  return splitted && splitted.length ? splitted : undefined;
};
var getHelpTopic = function getHelpTopic(str) {
  return splitStringOnFirst(str, ' ')[1] || 'help';
}; // Map empty input to :help help
var lowerCase = function lowerCase(str) {
  return str.toLowerCase();
};
var trim = function trim(str) {
  return str.trim();
};
var replaceSpaceWithDash = function replaceSpaceWithDash(str) {
  return str.replace(/\s/g, '-');
};
var snakeToCamel = function snakeToCamel(str) {
  return str.replace(/(-\w)/g, function (match) {
    return match[1].toUpperCase();
  });
};
var prependUnderscore = function prependUnderscore(str) {
  return '_' + str;
};
var transformCommandToHelpTopic = function transformCommandToHelpTopic(inputStr) {
  var res = [inputStr || ''].map(getHelpTopic).map(lowerCase).map(trim).map(replaceSpaceWithDash).map(snakeToCamel).map(prependUnderscore);
  return res[0];
};
// CONCATENATED MODULE: ./modules/Stream/HelpFrame.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */






var HelpFrame_HelpFrame = function HelpFrame(_ref) {
  var frame = _ref.frame;
  var help = 'Help topic not specified';
  if (frame.result) {
    help = /*#__PURE__*/react_default.a.createElement(Guides_Slide, {
      html: frame.result
    });
  } else {
    var helpTopic = transformCommandToHelpTopic(frame.cmd);
    if (helpTopic !== '') {
      var content = Help_html[helpTopic];
      if (content !== undefined) {
        help = /*#__PURE__*/react_default.a.createElement(Guides_Slide, {
          html: content
        });
      } else {
        help = /*#__PURE__*/react_default.a.createElement(Guides_Slide, {
          html: Help_html['_unfound']
        });
      }
    }
  }
  return /*#__PURE__*/react_default.a.createElement(Stream_FrameTemplate, {
    className: "helpFrame",
    header: frame,
    contents: help
  });
};
/* harmony default export */ var Stream_HelpFrame = (HelpFrame_HelpFrame);
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Typography/index.js
var Typography = __webpack_require__("../../node_modules/@material-ui/core/Typography/index.js");
var Typography_default = /*#__PURE__*/__webpack_require__.n(Typography);

// CONCATENATED MODULE: ./components/headers/Headers.jsx
function Headers_extends() { Headers_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Headers_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



var Headers_H3 = function H3(props) {
  return /*#__PURE__*/react_default.a.createElement(Typography_default.a, Headers_extends({
    variant: "h3",
    gutterBottom: true,
    component: "h3"
  }, props));
};
var Headers_H2 = function H2(props) {
  return /*#__PURE__*/react_default.a.createElement("h2", props);
};
var Headers_H1 = function H1(props) {
  return /*#__PURE__*/react_default.a.createElement("h1", props);
};
var Headers_H4 = function H4(props) {
  return /*#__PURE__*/react_default.a.createElement("h4", props);
};
var Headers_H5 = function H5(props) {
  return /*#__PURE__*/react_default.a.createElement("h5", props);
};
// CONCATENATED MODULE: ./components/headers/index.js
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



// CONCATENATED MODULE: ./components/Text.jsx
function Text_extends() { Text_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Text_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



var Text_Lead = function Lead(props) {
  return /*#__PURE__*/react_default.a.createElement(Typography_default.a, Text_extends({
    component: "p",
    variant: "body2"
  }, props));
};
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/CircularProgress/index.js
var CircularProgress = __webpack_require__("../../node_modules/@material-ui/core/CircularProgress/index.js");
var CircularProgress_default = /*#__PURE__*/__webpack_require__.n(CircularProgress);

// EXTERNAL MODULE: ./components/buttons/style.css
var buttons_style = __webpack_require__("./components/buttons/style.css");
var buttons_style_default = /*#__PURE__*/__webpack_require__.n(buttons_style);

// CONCATENATED MODULE: ./components/buttons/index.jsx
function buttons_typeof(obj) { "@babel/helpers - typeof"; return buttons_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, buttons_typeof(obj); }
var buttons_excluded = ["icon", "label", "children", "isLoading"],
  buttons_excluded2 = ["selected"],
  buttons_excluded3 = ["pressed", "children"],
  buttons_excluded4 = ["className"],
  buttons_excluded5 = ["children"];
function buttons_extends() { buttons_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return buttons_extends.apply(this, arguments); }
function buttons_objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = buttons_objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }
function buttons_objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }
function buttons_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function buttons_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function buttons_createClass(Constructor, protoProps, staticProps) { if (protoProps) buttons_defineProperties(Constructor.prototype, protoProps); if (staticProps) buttons_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function buttons_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) buttons_setPrototypeOf(subClass, superClass); }
function buttons_setPrototypeOf(o, p) { buttons_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return buttons_setPrototypeOf(o, p); }
function buttons_createSuper(Derived) { var hasNativeReflectConstruct = buttons_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = buttons_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = buttons_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return buttons_possibleConstructorReturn(this, result); }; }
function buttons_possibleConstructorReturn(self, call) { if (call && (buttons_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return buttons_assertThisInitialized(self); }
function buttons_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function buttons_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function buttons_getPrototypeOf(o) { buttons_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return buttons_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */






var buttons_ToolTip = /*#__PURE__*/function (_Component) {
  buttons_inherits(ToolTip, _Component);
  var _super = buttons_createSuper(ToolTip);
  function ToolTip(props) {
    var _this;
    buttons_classCallCheck(this, ToolTip);
    _this = _super.call(this, props);
    _this.state = {
      mouseover: false,
      id: props.id,
      icon: props.icon,
      name: props.tooltip,
      hidden: true
    };
    return _this;
  }
  buttons_createClass(ToolTip, [{
    key: "onMouseEnterHandler",
    value: function onMouseEnterHandler() {
      this.setState({
        mouseover: true
      });
    }
  }, {
    key: "onMouseLeaveHandler",
    value: function onMouseLeaveHandler() {
      this.setState({
        mouseover: false
      });
    }
  }, {
    key: "render",
    value: function render() {
      var tooltip = null;
      return /*#__PURE__*/react_default.a.createElement("div", {
        onMouseLeave: this.onMouseLeaveHandler.bind(this),
        onMouseEnter: this.onMouseEnterHandler.bind(this)
      }, this.props.children, tooltip);
    }
  }]);
  return ToolTip;
}(react["Component"]);
var buttons_CloseButton = function CloseButton(props) {
  return /*#__PURE__*/react_default.a.createElement("button", props, "\xD7");
};
var EditorButton = styled_components_es["default"].span.withConfig({
  displayName: "buttons__EditorButton",
  componentId: "sc-q5ir1y-0"
})(["font-family:", ";font-style:normal !important;font-weight:400 !important;font-variant:normal !important;text-transform:none !important;speak:none;-webkit-font-smoothing:antialiased;color:", ";background-color:", ";border-radius:50%;width:28px;height:28px;font-size:28px;line-height:28px;text-decoration:none;text-align:center;cursor:pointer;vertical-align:middle;display:inline-block;&:before{content:", ";text-decoration:none;}&:hover::before{content:", ";}"], function (props) {
  return props.theme.streamlineFontFamily;
}, function (props) {
  return props.theme.secondaryButtonText;
}, function (props) {
  return props.theme.secondaryButtonBackground;
}, function (props) {
  return props.icon;
}, function (props) {
  return props.hoverIcon;
});
var EditModeEditorButton = Object(styled_components_es["default"])(EditorButton).withConfig({
  displayName: "buttons__EditModeEditorButton",
  componentId: "sc-q5ir1y-1"
})(["color:", ";"], function (props) {
  return props.theme.editModeButtonText;
});
var buttons_FormButton = function FormButton(props) {
  var icon = props.icon,
    label = props.label,
    children = props.children,
    isLoading = props.isLoading,
    rest = buttons_objectWithoutProperties(props, buttons_excluded);
  var buttonChildren = children;
  if (icon && label) {
    buttonChildren = /*#__PURE__*/react_default.a.createElement(react_default.a.Fragment, null, label, " ", icon);
  } else if (icon) {
    buttonChildren = icon;
  } else if (label) {
    buttonChildren = label;
  }
  return /*#__PURE__*/react_default.a.createElement(Button_default.a, buttons_extends({
    type: "button",
    variant: "outlined",
    disabled: isLoading
  }, rest), buttonChildren, isLoading && /*#__PURE__*/react_default.a.createElement(CircularProgress_default.a, {
    size: 24,
    style: {
      position: 'absolute',
      top: '50%',
      left: '50%',
      marginTop: '-12px',
      marginLeft: '-12px'
    }
  }));
};
var buttons_CypherFrameButton = function CypherFrameButton(props) {
  var selected = props.selected,
    rest = buttons_objectWithoutProperties(props, buttons_excluded2);
  return selected ? /*#__PURE__*/react_default.a.createElement(StyledSelectedCypherFrameButton, rest) : /*#__PURE__*/react_default.a.createElement(StyledCypherFrameButton, rest);
};
var StyledCypherFrameButton = styled_components_es["default"].li.withConfig({
  displayName: "buttons__StyledCypherFrameButton",
  componentId: "sc-q5ir1y-2"
})(["color:", ";background-color:transparent;border-bottom:", ";height:58px;font-size:21px !important;line-height:21px;padding-top:12px;vertical-align:bottom;cursor:pointer;overflow:hidden;text-align:center;&:hover{background-color:", ";color:", ";fill:", ";text-decoration:none;}"], function (props) {
  return props.theme.secondaryButtonText;
}, function (props) {
  return props.theme.inFrameBorder;
}, function (props) {
  return props.theme.secondaryButtonBackgroundHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
});
var StyledSelectedCypherFrameButton = Object(styled_components_es["default"])(StyledCypherFrameButton).withConfig({
  displayName: "buttons__StyledSelectedCypherFrameButton",
  componentId: "sc-q5ir1y-3"
})(["background-color:", ";color:", ";fill:", ";"], function (props) {
  return props.theme.secondaryButtonBackgroundHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
});
var buttons_FrameButton = function FrameButton(props) {
  var pressed = props.pressed,
    children = props.children,
    rest = buttons_objectWithoutProperties(props, buttons_excluded3);
  return pressed ? /*#__PURE__*/react_default.a.createElement(StyledFrameButtonPressed, rest, children) : /*#__PURE__*/react_default.a.createElement(StyledFrameButton, rest, children);
};
var StyledFrameButton = styled_components_es["default"].li.withConfig({
  displayName: "buttons__StyledFrameButton",
  componentId: "sc-q5ir1y-4"
})(["color:", ";background-color:transparent;border-left:", ";height:", "px;width:41px;cursor:pointer;overflow:hidden;text-align:center;line-height:40px;display:inline-block;&:hover{background-color:", ";color:", ";fill:", ";text-decoration:none;}"], function (props) {
  return props.theme.secondaryButtonText;
}, function (props) {
  return props.theme.inFrameBorder;
}, dim.frameTitlebarHeight, function (props) {
  return props.theme.secondaryButtonBackgroundHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
});
var StyledFrameButtonPressed = Object(styled_components_es["default"])(StyledFrameButton).withConfig({
  displayName: "buttons__StyledFrameButtonPressed",
  componentId: "sc-q5ir1y-5"
})(["background-color:", ";color:", ";fill:", ";"], function (props) {
  return props.theme.secondaryButtonBackgroundHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
}, function (props) {
  return props.theme.secondaryButtonTextHover;
});
var DefaultA = styled_components_es["default"].a.withConfig({
  displayName: "buttons__DefaultA",
  componentId: "sc-q5ir1y-6"
})(["color:", ";&:hover{color:", ";text-decoration:none;}"], function (props) {
  return props.theme.secondaryButtonText;
}, function (props) {
  return props.theme.secondaryBackground;
});
var FrameButtonAChild = Object(styled_components_es["default"])(DefaultA).withConfig({
  displayName: "buttons__FrameButtonAChild",
  componentId: "sc-q5ir1y-7"
})(["display:block;text-decoration:none;&:focus,&:active,&:hover{outline:0;text-decoration:none;}"]);
var buttons_ActionButton = function ActionButton(props) {
  var className = props.className,
    rest = buttons_objectWithoutProperties(props, buttons_excluded4);
  return /*#__PURE__*/react_default.a.createElement("button", buttons_extends({
    className: className + ' ' + buttons_style_default.a.action
  }, rest));
};
var BaseCarouselButton = styled_components_es["default"].button.withConfig({
  displayName: "buttons__BaseCarouselButton",
  componentId: "sc-q5ir1y-8"
})(["order:1;background-color:rgb(34,34,34);border:3px solid rgb(255,255,255);color:rgb(255,255,255);cursor:pointer;margin:0 20px;font-family:", ";font-size:60px;font-weight:100;height:40px;min-height:40px;border-radius:50%;line-height:0;opacity:0.5;position:relative;text-align:center;text-decoration:none;text-shadow:rgba(0,0,0,0.6) 0px 1px 2px;top:130px;width:40px;min-width:40px;-webkit-font-smoothing:antialiased;user-select:none;&:hover{color:#fff;text-decoration:none;filter:alpha(opacity=90);outline:0;opacity:0.9;}&:focus{outline:none;}"], function (props) {
  return props.theme.primaryFontFamily;
});
var CarouselButtonOverlay = styled_components_es["default"].span.withConfig({
  displayName: "buttons__CarouselButtonOverlay",
  componentId: "sc-q5ir1y-9"
})(["position:absolute;top:13px;left:9px;"]);
var buttons_CarouselButton = function CarouselButton(props) {
  var children = props.children,
    rest = buttons_objectWithoutProperties(props, buttons_excluded5);
  return /*#__PURE__*/react_default.a.createElement(BaseCarouselButton, rest, /*#__PURE__*/react_default.a.createElement(CarouselButtonOverlay, null, children));
};
var StyledLink = styled_components_es["default"].a.withConfig({
  displayName: "buttons__StyledLink",
  componentId: "sc-q5ir1y-10"
})(["cursor:pointer;text-decoration:none;&:hover{color:#5dade2;text-decoration:none;}"]);
var SyncSignInButton = Object(styled_components_es["default"])(buttons_FormButton).withConfig({
  displayName: "buttons__SyncSignInButton",
  componentId: "sc-q5ir1y-11"
})(["background-color:#5fb434;color:#ffffff;&:hover{background-color:", ";color:#797979;}"], function (props) {
  return props.theme.formButtonBackgroundHover;
});
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/MenuItem/index.js
var MenuItem = __webpack_require__("../../node_modules/@material-ui/core/MenuItem/index.js");
var MenuItem_default = /*#__PURE__*/__webpack_require__.n(MenuItem);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Divider/index.js
var Divider = __webpack_require__("../../node_modules/@material-ui/core/Divider/index.js");
var Divider_default = /*#__PURE__*/__webpack_require__.n(Divider);

// CONCATENATED MODULE: ./services/EtlToolService.js
function _createForOfIteratorHelper(o, allowArrayLike) { var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"]; if (!it) { if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") { if (it) o = it; var i = 0; var F = function F() {}; return { s: F, n: function n() { if (i >= o.length) return { done: true }; return { done: false, value: o[i++] }; }, e: function e(_e) { throw _e; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var normalCompletion = true, didErr = false, err; return { s: function s() { it = it.call(o); }, n: function n() { var step = it.next(); normalCompletion = step.done; return step; }, e: function e(_e2) { didErr = true; err = _e2; }, f: function f() { try { if (!normalCompletion && it["return"] != null) it["return"](); } finally { if (didErr) throw err; } } }; }
function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }
function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }
function EtlToolService_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function EtlToolService_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function EtlToolService_createClass(Constructor, protoProps, staticProps) { if (protoProps) EtlToolService_defineProperties(Constructor.prototype, protoProps); if (staticProps) EtlToolService_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
/*
 * API interface with the ETL tool (java)
 */

var EtlToolService_JAR_NAME = 'dist/neo4j-etl.jar';
var TEST_CONNECTION_MAIN_CLASS = 'org.neo4j.etl.rdbms.Support';
var ETL_MAIN_CLASS = 'org.neo4j.etl.NeoIntegrationCli';
var ETL_WRITE_MAPPING_CLASS = 'org.neo4j.etl.io.Support';
var EtlToolService_ETL_FILE_UTILS_CLASS = 'org.neo4j.etl.util.FileUtils';
var ETL_DB_SUPPORT_CLASS = 'org.neo4j.etl.neo4j.support.DatabaseSupport';
var GRAPH_DB_FOLDER = 'graph.db/';
var DEFAULT_USING = 'cypher:batch';
var OPTIONS_FILE = 'import-tool-options.json';
var MAPPING_FILE = 'mapping.json';
var DEFAULT_UNWIND_BATCH_SIZE = 1000;
var DEFAULT_TX_BATCH_SIZE = 10000;
var IMPORT_MODES = [{
  name: 'Online direct import From SQL',
  cmd: 'cypher:fromSQL',
  mode: 'online'
}, {
  name: 'Online Import BATCH',
  cmd: 'cypher:batch',
  mode: 'online'
}, {
  name: 'Online Import LOAD CSV',
  cmd: 'cypher:direct',
  mode: 'online'
}, {
  name: 'Bulk Import',
  cmd: 'bulk:neo4j-import',
  mode: 'offline'
}];
var urlConnection = function urlConnection(type, host, port, database) {
  switch (type) {
    case 'oracle':
      // jdbc:oracle:thin:@localhost:1521/xe
      return "jdbc:oracle:thin:@".concat(host, ":").concat(port, ":").concat(database);
    case 'mysql':
      return "jdbc:mysql://".concat(host, ":").concat(port, "/").concat(database, "?autoReconnect=true&useSSL=false&useCursorFetch=true&allowPublicKeyRetrieval=true");
    case 'postgresql':
      return "jdbc:postgresql://".concat(host, ":").concat(port, "/").concat(database, "?ssl=false");
    case 'mssql':
      return "jdbc:sqlserver://".concat(host, ":").concat(port, ";databaseName=").concat(database);
    case 'db2':
      return "jdbc:db2://".concat(host, ":").concat(port, "/").concat(database);
    case 'jdbc':
      return "jdbc:xxxx://".concat(host, ":").concat(port, "/").concat(database);
    default:
      return '';
  }
};
var quotingCmdFunz = function quotingCmdFunz(arg) {
  return "\"" + arg + "\"";
};
var EtlToolService_EtlToolService = /*#__PURE__*/function () {
  /*
    * @param {configuration} host, port, database, username, password
    */

  function EtlToolService(configuration, importFolder) {
    EtlToolService_classCallCheck(this, EtlToolService);
    this.configuration = configuration;
    this.setImportMode();
    this.importFolder = importFolder;
    this.setUnwindBatchSize(DEFAULT_UNWIND_BATCH_SIZE);
    this.setTxBatchSize(DEFAULT_TX_BATCH_SIZE);
  }
  EtlToolService_createClass(EtlToolService, [{
    key: "setImportMode",
    value: function setImportMode(using) {
      this.using = using || DEFAULT_USING;
    }
  }, {
    key: "setUnwindBatchSize",
    value: function setUnwindBatchSize(size) {
      console.log('SET UNWIND BATCH SIZE TO: ' + size);
      this.unwindBatchSize = size;
    }
  }, {
    key: "setTxBatchSize",
    value: function setTxBatchSize(size) {
      console.log('SET TX BATCH SIZE TO: ' + size);
      this.txBatchSize = size;
    }
  }, {
    key: "setDatabaseName",
    value: function setDatabaseName(databaseName) {
      this.databaseName = databaseName;
    }
  }, {
    key: "setDbAutoCreation",
    value: function setDbAutoCreation(dbAutoCreation) {
      this.dbAutoCreation = dbAutoCreation || false;
    }
  }, {
    key: "isOnlineMode",
    value: function isOnlineMode(usingCommand) {
      var _iterator = _createForOfIteratorHelper(IMPORT_MODES),
        _step;
      try {
        for (_iterator.s(); !(_step = _iterator.n()).done;) {
          var mode = _step.value;
          if (mode.cmd === usingCommand) {
            return mode.mode === 'online';
          }
        }
      } catch (err) {
        _iterator.e(err);
      } finally {
        _iterator.f();
      }
    }
  }, {
    key: "getEtlHome",
    value: function getEtlHome() {
      return new URL(window.location.href).searchParams.get('neo4jDesktopGraphAppRootPath');
    }
  }, {
    key: "buildAbsolutePath",
    value: function buildAbsolutePath(relativePath) {
      return this.getEtlHome() + "/" + relativePath;
    }
  }, {
    key: "getJarPath",
    value: function getJarPath() {
      return this.buildAbsolutePath(EtlToolService_JAR_NAME);
    }
  }, {
    key: "buildPathToFile",
    value: function buildPathToFile() {
      if (this.configuration.schema !== undefined && this.configuration.schema !== null && this.configuration.schema !== "") {
        return this.configuration.driverType + "_" + this.configuration.database + "_" + this.configuration.schema + "_" + "".concat(MAPPING_FILE);
      } else {
        return this.configuration.driverType + "_" + this.configuration.database + "_" + "".concat(MAPPING_FILE);
      }
    }
  }, {
    key: "runJavaProcess",
    value: function runJavaProcess(parameters, processListener) {
      try {
        /* To manual test execution errors
        var runnable = confirm("Run Java?")
        if(! runnable){
          parameters.class = "manual_error"
        }
        */
        window.neo4jDesktopApi.executeJava(parameters).then(processListener.listener(), processListener.errListener);
      } catch (error) {
        processListener.fail(error);
      }
    }

    /*
      * @param {url} the JDBC connection url
      * @return array<string> with the connection arguments
      */
  }, {
    key: "buildJDBCConnectionArgs",
    value: function buildJDBCConnectionArgs(url) {
      var args = [];
      args.push(url);
      args.push(this.configuration.username);
      args.push(this.configuration.password);
      return args;
    } // buildConnectionArgs
  }, {
    key: "createConnectionURL",
    value: function createConnectionURL(type) {
      var url = urlConnection(type, this.configuration.host, this.configuration.port, this.configuration.database);
      if (this.configuration.connectionURL) {
        url = this.configuration.connectionURL;
      }
      return url;
    }
  }, {
    key: "testJDBCConnection",
    value: function testJDBCConnection(type, processListener) {
      var url = this.createConnectionURL(type);
      var parameters = {
        options: '',
        classpath: this.buildClasspath(),
        "class": TEST_CONNECTION_MAIN_CLASS,
        arguments: this.buildJDBCConnectionArgs(url)
      };
      this.runJavaProcess(parameters, processListener);
    } // testJDBCConnection
  }, {
    key: "buildGenerateMappingsArgs",
    value: function buildGenerateMappingsArgs(destinationGraph, type) {
      var quoting = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : function (x) {
        return x;
      };
      var args = [];
      args.push('generate-metadata-mapping');
      args.push('--rdbms:url');
      args.push(quoting(this.createConnectionURL(type)));
      args.push('--rdbms:password');
      args.push(quoting(this.configuration.password));
      args.push('--rdbms:user');
      args.push(quoting(this.configuration.username));
      if (this.configuration.schema !== undefined && this.configuration.schema !== '') {
        args.push('--schema');
        var schemaToAnalyze = this.configuration.schema;
        args.push(quoting(schemaToAnalyze));
      }
      args.push('--output-mapping-file');
      args.push(quoting("".concat(this.importFolder) + '/' + this.buildPathToFile()));
      return args;
    }
  }, {
    key: "buildClasspath",
    value: function buildClasspath() {
      var jarPath = this.getJarPath();
      return !this.configuration.path ? [jarPath] : [jarPath].concat(this.configuration.path);
    }
  }, {
    key: "runMapping",
    value: function runMapping(destinationGraph, type, processListener) {
      var parameters = {
        options: '',
        classpath: this.buildClasspath(),
        "class": ETL_MAIN_CLASS,
        arguments: this.buildGenerateMappingsArgs(destinationGraph, type)
      };
      this.runJavaProcess(parameters, processListener);
    }
  }, {
    key: "buildImportDataArgs",
    value: function buildImportDataArgs(destinationGraph, type, path, mappingFile) {
      var quoting = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : function (x) {
        return x;
      };
      var args = [];
      var isOnline = this.isOnlineMode(this.using);
      args.push('export');
      args.push('--mapping-file');
      args.push(quoting(mappingFile));
      // No destination if it's online
      if (!isOnline) {
        args.push('--destination');
        args.push(quoting("".concat(path, "/data/databases/").concat(GRAPH_DB_FOLDER)));
        args.push('--import-tool');
        args.push(quoting("".concat(path, "/bin")));
      }
      args.push('--rdbms:password');
      args.push(quoting(this.configuration.password));
      args.push('--rdbms:user');
      args.push(quoting(this.configuration.username));
      args.push('--rdbms:url');
      args.push(quoting(this.createConnectionURL(type)));
      if (this.using !== 'cypher:fromSQL') {
        args.push('--csv-directory');
        args.push(quoting(this.importFolder));
      }
      args.push('--options-file');
      args.push(quoting("".concat(this.importFolder, "/").concat(OPTIONS_FILE)));
      args.push('--using');
      args.push(quoting(this.using));
      if (this.using === 'cypher:batch' || this.using === 'cypher:fromSQL') {
        args.push('--unwind-batch-size');
        args.push(this.unwindBatchSize <= 0 ? DEFAULT_UNWIND_BATCH_SIZE : this.unwindBatchSize);
        args.push('--tx-batch-size');
        args.push(this.txBatchSize <= 0 ? DEFAULT_TX_BATCH_SIZE : this.txBatchSize);
      }
      var databaseName = this.databaseName || '';
      if (databaseName !== '') {
        args.push('--neo4j:database-name');
        args.push(databaseName);
      }
      var bolt = destinationGraph.connection.configuration.protocols.bolt;
      args.push('--neo4j:url');
      args.push(quoting(bolt.url || "bolt://".concat(bolt.host, ":").concat(bolt.port)));
      args.push('--neo4j:user');
      args.push(quoting(bolt.username || 'neo4j'));
      args.push('--neo4j:password');
      args.push(quoting(bolt.password || 'neo4j'));
      var driverOptions = this.getBoltDriverOptions(bolt);
      if (driverOptions !== '') {
        args.push('--neo4j:driver-options');
        args.push(btoa(driverOptions));
      }

      // No force mode if it's online
      if (!isOnline) {
        var autoCreation = this.dbAutoCreation || false;
        if (autoCreation) {
          args.push('--neo4j:db-auto-creation');
        }
        args.push('--force');
      }
      return args;
    }
  }, {
    key: "getBoltDriverOptions",
    value: function getBoltDriverOptions(bolt) {
      var driverOptions = {};
      var tls = bolt.tlsLevel || '';
      if (tls === 'REQUIRED') {
        driverOptions['encryption'] = true;
      }
      return Object.keys(driverOptions).length ? JSON.stringify(driverOptions) : '';
    }
  }, {
    key: "runImportData",
    value: function runImportData(destinationGraph, multilineOptions, type, processListener) {
      // Get Neo4j home
      var path = destinationGraph.connection.configuration.path;
      var pslFailOnly = new services_ProcessListener(function (out) {}, processListener.fail);

      // Write options file before running import
      try {
        var encodedString = btoa(JSON.stringify(multilineOptions));
        var _parameters = {
          options: '',
          classpath: [EtlToolService_JAR_NAME],
          "class": ETL_WRITE_MAPPING_CLASS,
          arguments: ['write', "".concat(this.importFolder, "/").concat(OPTIONS_FILE), encodedString, false]
        };
        this.runJavaProcess(_parameters, pslFailOnly);
      } catch (error) {
        processListener.fail(error);
      }
      var parameters = {
        options: '',
        classpath: this.buildClasspath(),
        "class": ETL_MAIN_CLASS,
        arguments: this.buildImportDataArgs(destinationGraph, type, path, this.importFolder + '/' + this.buildPathToFile())
      };
      this.runJavaProcess(parameters, processListener);
    }
  }, {
    key: "getDatabaseList",
    value: function getDatabaseList(destinationGraph, successCallback, errorCallback) {
      var bolt = destinationGraph.connection.configuration.protocols.bolt;
      errorCallback = errorCallback || function () {};
      var resultListener = new services_ProcessListener(function (out) {
        successCallback(JSON.parse(out.stdout));
      }, errorCallback);
      try {
        var parameters = {
          options: '',
          classpath: [EtlToolService_JAR_NAME],
          "class": ETL_DB_SUPPORT_CLASS,
          arguments: ['databaseList', bolt.url, bolt.username, bolt.password, btoa(this.getBoltDriverOptions(bolt))]
        };
        this.runJavaProcess(parameters, resultListener);
      } catch (error) {
        errorCallback(error);
      }
    }

    /*
       * Test if the configuration is a valid JDBC Connection
       * 
      */
  }, {
    key: "testConnection",
    value: function testConnection(processListener) {
      console.info('testConnection', this.configuration);
      this.testJDBCConnection(this.configuration.driverType, processListener);
    } // testConnection

    /**
     * Call the ETL generate mapping tool
     */
  }, {
    key: "generateMapping",
    value: function generateMapping(destinationGraph, processListener) {
      this.runMapping(destinationGraph, this.configuration.driverType, processListener);
    } // generateMapping
  }, {
    key: "writeChunk",
    value: function writeChunk(path, chunks, part, onSuccess, onFailure) {
      var self = this;
      var processListener = new services_ProcessListener();
      processListener.setContext({
        path: path,
        chunks: chunks,
        part: part
      });
      processListener.setSuccessHandler(function (_ref) {
        var stdout = _ref.stdout,
          stderr = _ref.stderr;
        var context = processListener.getContext();
        if (context.part < context.chunks.length - 1) {
          self.writeChunk(context.path, context.chunks, context.part + 1, onSuccess, onFailure);
        } else {
          onSuccess();
        }
      });
      processListener.setFailureHandler(function (error) {
        onFailure(error);
      });
      var filename = this.buildPathToFile();
      var parameters = {
        options: '',
        classpath: [EtlToolService_JAR_NAME],
        "class": ETL_WRITE_MAPPING_CLASS,
        // if it's the first part no append mode otherwise append mode
        arguments: ['write', this.importFolder + '/' + filename, chunks[part], part === 0 ? false : true]
      };
      this.runJavaProcess(parameters, processListener);
    }
  }, {
    key: "writeMapping",
    value: function writeMapping(destinationGraph, mapping, resultCallback, fail) {
      var BUFFER_SIZE = 14000;
      var encodedString = btoa(JSON.stringify(mapping));
      var path = ''; //destinationGraph.connection.configuration.path

      var intParts = Math.floor(encodedString.length / BUFFER_SIZE);
      var chunks = [];
      for (var part = 0; part < intParts; part++) {
        var partString = encodedString.substring(part * BUFFER_SIZE, (part + 1) * BUFFER_SIZE);
        chunks = chunks.concat(partString);
      }
      chunks = chunks.concat(encodedString.substring(intParts * BUFFER_SIZE));
      this.writeChunk(path, chunks, 0, resultCallback, fail);
    }
  }, {
    key: "readMapping",
    value: function readMapping(destinationGraph, resultCallback) {
      var fail = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : function () {};
      var resultListener = new services_ProcessListener(function (out) {
        resultCallback(JSON.parse(out.stdout));
      }, fail);
      try {
        var filePath = "".concat(this.importFolder, "/").concat(this.buildPathToFile());
        var parameters = {
          options: '',
          classpath: [EtlToolService_JAR_NAME],
          "class": EtlToolService_ETL_FILE_UTILS_CLASS,
          arguments: ['readfile', filePath]
        };
        this.runJavaProcess(parameters, resultListener);
      } catch (error) {
        fail(error);
      }
    }
  }, {
    key: "importData",
    value: function importData(destinationGraph, multilineOptions, processListener) {
      this.runImportData(destinationGraph, multilineOptions, this.configuration.driverType, processListener);
    }
  }, {
    key: "buildShellCommand",
    value: function buildShellCommand(etlArgs) {
      var cpSeparator = window.navigator.userAgent.indexOf("Windows") != -1 ? ";" : ":";
      var cp = this.buildClasspath().join(cpSeparator);
      var javaArgs = ["java -cp", quotingCmdFunz(cp), ETL_MAIN_CLASS];
      var allArgs = javaArgs.concat(etlArgs);
      var cmd = allArgs.join(" ");
      return cmd;
    }
  }, {
    key: "buildShellCommandOfMapping",
    value: function buildShellCommandOfMapping(currentGraph) {
      var etlArgs = this.buildGenerateMappingsArgs(currentGraph, this.configuration.driverType, quotingCmdFunz);
      return this.buildShellCommand(etlArgs);
    }
  }, {
    key: "buildShellCommandOfImport",
    value: function buildShellCommandOfImport(currentGraph) {
      var etlArgs = this.buildImportDataArgs(currentGraph, this.configuration.driverType, currentGraph.connection.configuration.path, this.importFolder + '/' + this.buildPathToFile(), quotingCmdFunz);
      return this.buildShellCommand(etlArgs);
    }
  }]);
  return EtlToolService;
}(); // EtlToolService
/* harmony default export */ var services_EtlToolService = (EtlToolService_EtlToolService);
// CONCATENATED MODULE: ./services/EtlMappingToolColors.js

var colors = [{
  background: '#FFE081',
  border: '#9AA1AC'
}, {
  background: '#C990C0',
  border: '#b261a5'
}, {
  background: '#F79767',
  border: '#f36924'
}, {
  background: '#57C7E3',
  border: '#23b3d7'
}, {
  background: '#F16667',
  border: '#eb2728'
}, {
  background: '#D9C8AE',
  border: '#c0a378'
}, {
  background: '#8DCC93',
  border: '#5db665'
}, {
  background: '#ECB5C9',
  border: '#da7298'
}, {
  background: '#4C8EDA',
  border: '#2870c2'
}, {
  background: '#FFC454',
  border: '#d7a013'
}, {
  background: '#DA7194',
  border: '#cc3c6c'
}, {
  background: '#569480',
  border: '#447666'
}];
function getRandomColor() {
  return lodash_default.a.sample(colors);
}
// CONCATENATED MODULE: ./services/EtlMappingTool.js
function EtlMappingTool_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function EtlMappingTool_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function EtlMappingTool_createClass(Constructor, protoProps, staticProps) { if (protoProps) EtlMappingTool_defineProperties(Constructor.prototype, protoProps); if (staticProps) EtlMappingTool_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
/**
 * Tool to explore and update the mapping JSON
 */



/**
 * Utility for SQL replacement. It changes the value in SELECT that represents the LABEL/RELATIONSHIP
 */
var buildSqlReplacer = function buildSqlReplacer(sql, colName, type) {
  var fromIndex = sql.indexOf("FROM");
  var selectSql = sql.substr(0, fromIndex);
  var fromSql = sql.substr(fromIndex);
  var cols = selectSql.split(",").map(function (c) {
    var newCol = {
      original: c,
      sqlCol: c,
      update: function update(newValue) {
        if (newCol.original.includes(type)) {
          newCol.sqlCol = newCol.original.replace(colName, newValue);
        }
      } //update
    };

    return newCol;
  });
  var res = {
    sql: sql,
    colName: colName,
    fromSql: fromSql,
    columns: cols,
    replaceType: function replaceType(newType) {
      res.columns.forEach(function (c) {
        return c.update(newType);
      });
      return res.columns.map(function (c) {
        return c.sqlCol;
      }).join(',') + fromSql;
    }
  };
  return res;
};

/**
 * Tool for navigate and update the JSON mapping returned by the backend
 */
var EtlMappingTool_EtlMappingTool = /*#__PURE__*/function () {
  function EtlMappingTool(originalMapping) {
    EtlMappingTool_classCallCheck(this, EtlMappingTool);
    this.originalMapping = originalMapping;
    var nodes = [];
    var rels = [];
    if (originalMapping) {
      nodes = this.originalMapping.filter(function (map) {
        return map['graph-object-type'] === 'Node';
      }).map(function (node) {
        return node.mappings.filter(function (m) {
          return m.field.type === 'Label';
        }).map(function (m) {
          m.column.name = m.column.name.replace(/`/g, '');
          var sqlReplacer = buildSqlReplacer(node.sql, m.column.name, "_NODE_LABEL_");
          var newNode = {
            nodeKey: (m.column.schema + '.' + m.column.table).toLowerCase(),
            label: m.column.name,
            tableName: m.column.table,
            columns: [],
            sqlReplacer: sqlReplacer,
            // link data
            original: node,
            // methods
            setLabel: function setLabel(newLabel) {
              m.column.name = newLabel;
              newNode.label = newLabel;
              node.sql = sqlReplacer.replaceType(newLabel);
            },
            setColumns: function setColumns(columns) {
              newNode.columns = columns;
            },
            color: getRandomColor()
          };
          return newNode;
        });
      }).map(function (m) {
        return m[0];
      });

      // add column base info
      nodes.forEach(function (node) {
        node.original.mappings.filter(function (m) {
          return m.column.type === 'SimpleColumn' && m.field.type != 'Label';
        }).forEach(function (m) {
          var newCol = {
            columnName: m.column.name,
            fieldName: m.field.name,
            sqlType: m.column['sql-data-type'],
            neo4jType: m.field['neo4j-data-type'],
            skip: m.skip,
            // methods
            setFieldName: function setFieldName(newName) {
              m.field.name = newName;
              newCol.fieldName = newName;
            },
            setSqlType: function setSqlType(newType) {
              m.column['sql-data-type'] = newType;
              newCol.sqlType = newType;
            },
            setNeoType: function setNeoType(newType) {
              m.field['neo4j-data-type'] = newType;
              newCol.neo4jType = newType;
            },
            setSkip: function setSkip(newValue) {
              m.skip = newValue;
              newCol.skip = newValue;
            }
          };
          node.columns.push(newCol);
        });
      }); // node

      // key info ((table,name) pk, fk)
      var keysInfo = nodes.map(function (node) {
        return node.original.mappings.filter(function (m) {
          return m.column.type === 'CompositeColumn' && m.field.type !== 'Label';
        }).map(function (m) {
          return {
            primaryKey: m.column.role === 'PrimaryKey',
            foreignKey: m.column.role === 'ForeignKey',
            columns: m.column.columns
          };
        }).map(function (m) {
          return m.columns.map(function (c) {
            return {
              primaryKey: m.primaryKey,
              foreignKey: m.foreignKey,
              tableName: c.table,
              columnName: c.name
            };
          });
        });
      }).map(function (node) {
        return node[0];
      }).reduce(function (prev, curr) {
        return prev.concat(curr);
      }, []);

      // add column keys info
      nodes.forEach(function (node) {
        node.columns.forEach(function (col) {
          var key = keysInfo.filter(function (key) {
            return key && key.tableName === node.tableName && key.columnName === col.columnName;
          });
          if (key && key[0]) {
            col.primaryKey = key[0].primaryKey;
            col.foreignKey = key[0].foreignKey;
            col.skip = false;
          }
        });
      });

      // RELATIONSHIP
      var getNodeKey = function getNodeKey(map, fieldType) {
        return map.mappings.filter(function (o) {
          return o.field.type === fieldType;
        }).map(function (s) {
          return s.field['id-space'];
        })[0];
      };
      var getNodeByKey = function getNodeByKey(nodeKey) {
        return nodes.filter(function (n) {
          return n.nodeKey === nodeKey;
        })[0];
      };
      var getRelCols = function getRelCols(map) {
        return map.mappings.filter(function (m) {
          return m.field.type === 'Data';
        }).map(function (m) {
          var newCol = {
            columnName: m.column.name,
            fieldName: m.field.name,
            sqlType: m.column["sql-data-type"],
            neo4jType: m.field["neo4j-data-type"],
            original: m,
            skip: m.skip,
            //methods
            setFieldName: function setFieldName(newName) {
              m.field.name = newName;
              newCol.fieldName = newName;
            },
            setSqlType: function setSqlType(newType) {
              m.column['sql-data-type'] = newType;
              newCol.sqlType = newType;
            },
            setNeoType: function setNeoType(newType) {
              m.field['neo4j-data-type'] = newType;
              newCol.neo4jType = newType;
            },
            setSkip: function setSkip(newValue) {
              m.skip = newValue;
              newCol.skip = newValue;
            }
          };
          return newCol;
        });
      };
      rels = this.originalMapping.filter(function (map) {
        return map['graph-object-type'] === 'Relationship';
      }).map(function (map) {
        return map.mappings.filter(function (m) {
          return m.field.type === 'RelationshipType';
        }).map(function (m) {
          var startRef = getNodeKey(map, 'StartId');
          var startNode = getNodeByKey(startRef);
          var endRef = getNodeKey(map, 'EndId');
          var endNode = getNodeByKey(endRef);
          var salt = Math.floor(Math.random() * 100 + 1);
          var hashKey = startRef + '_' + m.column.name + '_' + endRef + '_' + salt;
          hashKey = btoa(hashKey);
          var cols = getRelCols(map);
          m.column.name = m.column.name.replace(/`/g, '');
          var sqlReplacer = buildSqlReplacer(map.sql, m.column.name, "_RELATIONSHIP_TYPE_");
          var newRel = {
            relKey: hashKey,
            type: m.column.name,
            columns: cols,
            original: map,
            start: startRef,
            startNode: startNode,
            end: endRef,
            endNode: endNode,
            sqlReplacer: sqlReplacer,
            // methods
            setType: function setType(newType) {
              m.column.name = newType;
              newRel.type = newType;
              map.sql = newRel.sqlReplacer.replaceType(newType);
            },
            setColumns: function setColumns(columns) {
              newRel.columns = columns;
            }
          };
          return newRel;
        });
      }).map(function (m) {
        return m[0];
      });
      var cols = this.originalMapping.filter(function (map) {
        return map['graph-object-type'] === 'Relationship';
      }).map(function (map) {
        return map.mappings.filter(function (m) {
          return m.field.type === 'Data';
        }).map(function (m) {
          return {
            columnName: m.column.name,
            fieldName: m.field.name,
            sqlType: m.column['sql-data-type'],
            neo4jType: m.field['neo4j-data-type']
          };
        });
      }).map(function (m) {
        return m[0];
      });
    }
    this.graph = {
      nodes: nodes,
      relationships: rels
    };
  } // constructor
  EtlMappingTool_createClass(EtlMappingTool, [{
    key: "getAllNodes",
    value: function getAllNodes() {
      return this.graph.nodes;
    }
  }, {
    key: "getAllRelationships",
    value: function getAllRelationships() {
      return this.graph.relationships;
    }
  }]);
  return EtlMappingTool;
}(); // EtlMappingTool
var getMetricsMappingsDataByType = function getMetricsMappingsDataByType(mappings, type) {
  var types = mappings.filter(function (elem) {
    return elem['graph-object-type'] === type;
  });
  var count = types.length;
  var totalColumnCount = types.flatMap(function (elem) {
    return elem.mappings;
  }).length;
  var sqlTypeCount = types.flatMap(function (elem) {
    return elem.mappings;
  }).map(function (elem) {
    return elem.column;
  }).filter(function (elem) {
    return 'sql-data-type' in elem;
  }).reduce(function (accumulator, currentValue) {
    var sqlType = currentValue['sql-data-type'];
    accumulator["".concat(type, "-").concat(sqlType)] = (accumulator[type] || 0) + 1;
    return accumulator;
  }, {});
  sqlTypeCount["".concat(type, "-count")] = count;
  sqlTypeCount["".concat(type, "-totalColumnCount")] = totalColumnCount;
  return sqlTypeCount;
};
/* harmony default export */ var services_EtlMappingTool = (EtlMappingTool_EtlMappingTool);
// CONCATENATED MODULE: ./modules/Stream/mapping/StartMappingSlide.jsx
function StartMappingSlide_typeof(obj) { "@babel/helpers - typeof"; return StartMappingSlide_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, StartMappingSlide_typeof(obj); }
function StartMappingSlide_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function StartMappingSlide_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? StartMappingSlide_ownKeys(Object(source), !0).forEach(function (key) { StartMappingSlide_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : StartMappingSlide_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function StartMappingSlide_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function StartMappingSlide_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function StartMappingSlide_createClass(Constructor, protoProps, staticProps) { if (protoProps) StartMappingSlide_defineProperties(Constructor.prototype, protoProps); if (staticProps) StartMappingSlide_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function StartMappingSlide_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) StartMappingSlide_setPrototypeOf(subClass, superClass); }
function StartMappingSlide_setPrototypeOf(o, p) { StartMappingSlide_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return StartMappingSlide_setPrototypeOf(o, p); }
function StartMappingSlide_createSuper(Derived) { var hasNativeReflectConstruct = StartMappingSlide_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = StartMappingSlide_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = StartMappingSlide_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return StartMappingSlide_possibleConstructorReturn(this, result); }; }
function StartMappingSlide_possibleConstructorReturn(self, call) { if (call && (StartMappingSlide_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return StartMappingSlide_assertThisInitialized(self); }
function StartMappingSlide_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function StartMappingSlide_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function StartMappingSlide_getPrototypeOf(o) { StartMappingSlide_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return StartMappingSlide_getPrototypeOf(o); }
function StartMappingSlide_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

















var StartMappingSlide_StartMappingSlide = /*#__PURE__*/function (_Component) {
  StartMappingSlide_inherits(StartMappingSlide, _Component);
  var _super = StartMappingSlide_createSuper(StartMappingSlide);
  function StartMappingSlide(props) {
    var _this;
    StartMappingSlide_classCallCheck(this, StartMappingSlide);
    _this = _super.call(this, props);
    StartMappingSlide_defineProperty(StartMappingSlide_assertThisInitialized(_this), "startProjectState", function (selectedProject, connectionName) {
      return {
        selectedProject: selectedProject,
        connectionName: connectionName,
        waitingMessage: '',
        mappingDone: false,
        mappingStarted: false,
        selectedGraph: null
      };
    });
    StartMappingSlide_defineProperty(StartMappingSlide_assertThisInitialized(_this), "onConnectionClick", function (connectionInfo) {
      if (_this.state.mappingStarted) {
        return null;
      }
      _this.setState({
        connectionName: connectionInfo.name,
        mappingDone: false
      });
      _this.props.projectStore.currentProject.setCurrentSourceConnection(connectionInfo);
      _this.props.bus.send(CONNECTION_SAVED, {
        connectionName: connectionInfo.name
      });
    });
    StartMappingSlide_defineProperty(StartMappingSlide_assertThisInitialized(_this), "onRemoveConnection", function (connectionInfo) {
      _this.props.projectStore.currentProject.removeSourceConnection(connectionInfo);
      if (connectionInfo.name == _this.state.connectionName) {
        _this.props.projectStore.currentProject.setCurrentSourceConnection(null);
        _this.setState({
          connectionName: null
        });
      }
      _this.forceUpdate();
    });
    var currentProject = props.projectStore.currentProject;
    var _selectedProject = '';
    var selectedConnection = null;
    if (!lodash_default.a.isEmpty(currentProject)) {
      _selectedProject = currentProject.original.id;
      var currentConn = currentProject.getCurrentSourceConnection();
      var connectionInfo = props.connectionInfo;
      if (currentConn && connectionInfo.key && currentConn.key === connectionInfo.key) {
        selectedConnection = connectionInfo.name;
        currentProject.setCurrentSourceConnection(connectionInfo);
      }
    }
    _this.state = StartMappingSlide_objectSpread(StartMappingSlide_objectSpread({}, _this.startProjectState(_selectedProject, selectedConnection)), {}, {
      showLogs: false,
      output: ''
    });
    _this.selectProject = _this.selectProject.bind(StartMappingSlide_assertThisInitialized(_this));
    _this.toggleLogs = _this.toggleLogs.bind(StartMappingSlide_assertThisInitialized(_this));
    return _this;
  }
  StartMappingSlide_createClass(StartMappingSlide, [{
    key: "callMapping",
    value: function callMapping() {
      var _this2 = this;
      var myBus = this.props.bus;
      var connectionInfo = this.props.projectStore.currentProject.getCurrentSourceConnection() || {};
      var currentGraph = this.props.projectStore.currentProject.getCurrentGraph() || {};
      var importFolder = this.props.projectStore.currentProject.getImportFolder() || '';
      console.log('connection', connectionInfo);
      console.log('selectedGraph', currentGraph);
      this.setState({
        waitingMessage: "Loading metadata from ".concat(connectionInfo.name, " (").concat(connectionInfo.database, ")")
      });
      this.etl = new services_EtlToolService(connectionInfo, importFolder);
      var cmd = "COMMAND: " + this.etl.buildShellCommandOfMapping(currentGraph);
      console.info(cmd + "\nLOG:\n");
      this.setState({
        output: cmd,
        waitingMessage: 'Calling...',
        mappingDone: false,
        mappingStarted: true
      });
      var processListener = new services_ProcessListener();
      processListener.setErrorListener(function (out) {
        var outAppend = _this2.state.output + out;
        _this2.setState({
          output: outAppend,
          waitingMessage: 'Mapping...',
          mappingDone: false
        });
        if (outAppend.indexOf('Java is restricted to execute') !== -1) {
          myBus.send(JAVA_PERMISSION_ERROR_SHOW);
        }
      });
      var start = new Date().getTime();
      processListener.setSuccessHandler(function (_ref) {
        var stdout = _ref.stdout,
          stderr = _ref.stderr;
        var msg = showInfoMessage('Mapping successful');
        var outMsg = cmd + "\n" + stderr;
        _this2.etl.readMapping(currentGraph, function (mappings) {
          var mappingMetrics = StartMappingSlide_objectSpread(StartMappingSlide_objectSpread({}, getMetricsMappingsDataByType(mappings, 'Node')), getMetricsMappingsDataByType(mappings, 'Relationship'));
          mappingMetrics.elapsed = new Date().getTime() - start;
          console.log('Sending %o to metrics', mappingMetrics);
          window.neo4jDesktopApi.sendMetrics('neo4j-etl', 'mapping', mappingMetrics);
          var mappingTool = new services_EtlMappingTool(mappings);
          _this2.setState({
            output: outMsg,
            //out,
            waitingMessage: '',
            mappingDone: true,
            mappingStarted: false
          });
          myBus.send(msg.type, msg);
          myBus.send(MAPPING_DONE, {
            nodes: mappingTool.getAllNodes(),
            relationships: mappingTool.getAllRelationships(),
            mappings: mappings,
            connectionName: _this2.state.connectionName
          });
        }, function (error) {
          var actErr = showErrorMessage(error);
          myBus.send(actErr.type, actErr);
        });
      }); //processListener.setSuccessHandler

      processListener.setFailureHandler(function (error) {
        console.error(error);
        var actErr = showErrorMessage('Mapping error');
        myBus.send(actErr.type, actErr);
        _this2.setState({
          waitingMessage: '',
          output: cmd + "\n" + error,
          mappingStarted: false
        });
      }); //processListener.setFailureHandler        

      this.etl.generateMapping(currentGraph, processListener);
    }
  }, {
    key: "onGraphClick",
    value: function onGraphClick(graph) {
      console.info(graph);
      this.setState({
        selectedGraph: graph
      });
      this.props.projectStore.currentProject.setCurrentGraph(graph);
    }
  }, {
    key: "addConnection",
    value: function addConnection(e) {
      e.preventDefault();
      // Remove current frame
      this.props.bus.send(REMOVE, {
        id: 'MAPPING'
      });

      // Add connection frame
      this.props.bus.send(ADD, {
        state: {
          id: 'RDBMS',
          type: 'connection'
        }
      });
    }
  }, {
    key: "componentDidUpdate",
    value: function componentDidUpdate(prevProps) {
      var currentProject = this.props.projectStore.currentProject;
      var prevCurrentProject = prevProps.projectStore.currentProject;
      if (!lodash_default.a.isEmpty(currentProject) && currentProject.original !== prevCurrentProject.original) {
        this.setState({
          selectedProject: currentProject.original.id
        });
      }
    }
  }, {
    key: "selectProject",
    value: function selectProject(e) {
      var projectId = e.target.value;
      var project = lodash_default.a.find(this.props.projectStore.projects, {
        original: {
          id: projectId
        }
      });
      if (!lodash_default.a.isEmpty(project)) {
        this.setState(this.startProjectState(project, (project.getCurrentSourceConnection() || {}).name));
        this.props.bus.send(PROJECT_SELECTED, {
          currentProject: project
        });
      }
    }
  }, {
    key: "toggleLogs",
    value: function toggleLogs() {
      this.setState(function (state) {
        return {
          showLogs: !state.showLogs
        };
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this3 = this;
      if (lodash_default.a.isEmpty(this.props.projectStore.currentProject)) {
        return null;
      }
      var connections = this.props.projectStore.currentProject.getAllSourceConnections();
      var connection = this.props.projectStore.currentProject.getCurrentSourceConnection();
      var graphAvailable = this.props.projectStore.currentProject.getAllGraphs().length > 0;
      var graphs = this.props.projectStore.currentProject.getAllGraphs();
      var currentGraph = this.props.projectStore.currentProject.getCurrentGraph();
      var nextButtonEnabled = connection && this.state.mappingDone && !!currentGraph;
      return /*#__PURE__*/react_default.a.createElement(styled_StyledFrameSlide, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameTitleBar, null, /*#__PURE__*/react_default.a.createElement(Headers_H3, null, "Load Data Model"))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameMainSection, {
        style: {
          height: '100%',
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(Text_Lead, {
        component: "div"
      }, "Select the project:", " ", /*#__PURE__*/react_default.a.createElement(Select_default.a, {
        value: this.state.selectedProject,
        onChange: this.selectProject,
        disabled: this.state.mappingStarted
      }, this.props.projectStore.projects.map(function (project, index) {
        return /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          key: project.original.id,
          value: project.original.id
        }, project.original.name);
      }))), !graphAvailable && /*#__PURE__*/react_default.a.createElement(GraphNotAvailable, null, "The project selected don't have a Graph, please add one first."))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 5
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameMainSection, {
        style: {
          height: '100%',
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(Text_Lead, null, "Select what connection you want to import from or add a new one."), /*#__PURE__*/react_default.a.createElement(Text_Lead, null, "Selected RDBMS instance ", /*#__PURE__*/react_default.a.createElement("b", null, this.state.connectionName || (connection || {}).name || 'NONE')), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          width: '100%',
          height: '250px',
          overflowY: 'auto',
          paddingTop: '16px'
        }
      }, connections.map(function (conn) {
        return /*#__PURE__*/react_default.a.createElement(StyledGraphChip, {
          key: conn.key,
          icon: /*#__PURE__*/react_default.a.createElement(StyledChipIcon, {
            className: 'sl sl-database'
          }),
          onClick: _this3.onConnectionClick.bind(_this3, conn),
          connection: conn,
          dataId: conn.key,
          onDelete: _this3.onRemoveConnection.bind(_this3, conn),
          isSelected: connection && connection.key === conn.key,
          allowEdit: true,
          disabled: _this3.state.mappingStarted
        }, conn.name, /*#__PURE__*/react_default.a.createElement("br", null), "Database: ", conn.database && conn.database.toUpperCase(), /*#__PURE__*/react_default.a.createElement("br", null), "Type: ", conn.driverType && conn.driverType.toUpperCase());
      })))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 2,
        style: {
          textAlign: 'center'
        }
      }, /*#__PURE__*/react_default.a.createElement(Icon_default.a, {
        style: {
          fontSize: '100px'
        }
      }, "arrow_right_alt")), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 5
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameMainSection, {
        style: {
          height: '100%',
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(Text_Lead, null, "Select what Neo4j instance you want to import to"), /*#__PURE__*/react_default.a.createElement(Text_Lead, null, "Selected Neo4j instance ", /*#__PURE__*/react_default.a.createElement("b", null, (this.state.selectedGraph || currentGraph || {}).name || 'NONE')), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          width: '100%',
          height: '250px',
          overflowY: 'scroll'
        }
      }, graphs.map(function (graph, i) {
        return /*#__PURE__*/react_default.a.createElement(StyledGraphChip, {
          key: i,
          onClick: _this3.onGraphClick.bind(_this3, graph),
          "play-topic": "concepts",
          isSelected: !!currentGraph && currentGraph.id === graph.id,
          icon: /*#__PURE__*/react_default.a.createElement(StyledChipIcon, {
            className: 'nw nw-neo4j-outline-32px'
          })
        }, graph.name, /*#__PURE__*/react_default.a.createElement("br", null), graph.connection.databaseStatus, /*#__PURE__*/react_default.a.createElement("br", null), graph.connection.info.version, " - ", graph.connection.info.edition, /*#__PURE__*/react_default.a.createElement("br", null));
      })))), this.state.showLogs && /*#__PURE__*/react_default.a.createElement(react_default.a.Fragment, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(Divider_default.a, null)), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          height: '100%',
          overflowY: 'auto',
          maxHeight: '448px'
        }
      }, /*#__PURE__*/react_default.a.createElement("pre", null, this.state.output), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          textAlign: 'center',
          fontSize: '18px',
          marginTop: '20px'
        }
      }, this.state.waitingMessage)))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(Divider_default.a, null)), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.addConnection.bind(this),
        disabled: this.state.mappingStarted,
        label: "Add Connection"
      }), " ", this.state.output && /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.toggleLogs,
        label: "".concat(this.state.showLogs ? 'Hide' : 'See', " Logs")
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6,
        style: {
          textAlign: 'right'
        }
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        style: {
          marginRight: '8px'
        },
        onClick: this.callMapping.bind(this),
        isLoading: this.state.mappingStarted,
        disabled: this.state.mappingDone || this.state.mappingStarted || !connection || !graphAvailable,
        label: "Start Mapping"
      }), /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.props.next,
        disabled: !nextButtonEnabled,
        label: "Next"
      })));
    }
  }]);
  return StartMappingSlide;
}(react["Component"]);
var StartMappingSlide_mapStateToProps = function mapStateToProps(state) {
  var mapping = state.mapping;
  return {
    connectionInfo: mapping.connectionInfo || {},
    projectStore: state.projectStore
  };
};
/* harmony default export */ var mapping_StartMappingSlide = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(StartMappingSlide_mapStateToProps, null)(StartMappingSlide_StartMappingSlide)));
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Input/index.js
var Input = __webpack_require__("../../node_modules/@material-ui/core/Input/index.js");
var Input_default = /*#__PURE__*/__webpack_require__.n(Input);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Checkbox/index.js
var Checkbox = __webpack_require__("../../node_modules/@material-ui/core/Checkbox/index.js");
var Checkbox_default = /*#__PURE__*/__webpack_require__.n(Checkbox);

// CONCATENATED MODULE: ./components/Form.jsx
function Form_extends() { Form_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Form_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var TextInput = Object(styles["withStyles"])({})(Input_default.a);
var Form_CheckboxSelector = function CheckboxSelector(props) {
  return /*#__PURE__*/react_default.a.createElement(Checkbox_default.a, Form_extends({
    color: "primary"
  }, props));
};
// CONCATENATED MODULE: ./components/ConfirmDialog.jsx
function ConfirmDialog_typeof(obj) { "@babel/helpers - typeof"; return ConfirmDialog_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ConfirmDialog_typeof(obj); }
function ConfirmDialog_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ConfirmDialog_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ConfirmDialog_createClass(Constructor, protoProps, staticProps) { if (protoProps) ConfirmDialog_defineProperties(Constructor.prototype, protoProps); if (staticProps) ConfirmDialog_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ConfirmDialog_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ConfirmDialog_setPrototypeOf(subClass, superClass); }
function ConfirmDialog_setPrototypeOf(o, p) { ConfirmDialog_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ConfirmDialog_setPrototypeOf(o, p); }
function ConfirmDialog_createSuper(Derived) { var hasNativeReflectConstruct = ConfirmDialog_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ConfirmDialog_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ConfirmDialog_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ConfirmDialog_possibleConstructorReturn(this, result); }; }
function ConfirmDialog_possibleConstructorReturn(self, call) { if (call && (ConfirmDialog_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ConfirmDialog_assertThisInitialized(self); }
function ConfirmDialog_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ConfirmDialog_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ConfirmDialog_getPrototypeOf(o) { ConfirmDialog_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ConfirmDialog_getPrototypeOf(o); }


var ConfirmDialog_ConfirmDialog = /*#__PURE__*/function (_React$Component) {
  ConfirmDialog_inherits(ConfirmDialog, _React$Component);
  var _super = ConfirmDialog_createSuper(ConfirmDialog);
  function ConfirmDialog(props) {
    var _this;
    ConfirmDialog_classCallCheck(this, ConfirmDialog);
    _this = _super.call(this, props);
    _this.state = {
      open: false
    };
    _this.handleOpen = _this.handleOpen.bind(ConfirmDialog_assertThisInitialized(_this));
    _this.handleClose = _this.handleClose.bind(ConfirmDialog_assertThisInitialized(_this));
    _this.handleConfirm = _this.handleConfirm.bind(ConfirmDialog_assertThisInitialized(_this));
    return _this;
  }
  ConfirmDialog_createClass(ConfirmDialog, [{
    key: "handleOpen",
    value: function handleOpen() {
      this.setState({
        open: true
      });
    }
  }, {
    key: "handleClose",
    value: function handleClose() {
      this.setState({
        open: false
      });
    }
  }, {
    key: "handleConfirm",
    value: function handleConfirm() {
      this.props.onConfirm();
      this.handleClose();
    }
  }, {
    key: "render",
    value: function render() {
      var ButtonComponent = this.props.buttonComponent;
      return /*#__PURE__*/react_default.a.createElement(react_default.a.Fragment, null, /*#__PURE__*/react_default.a.createElement(ButtonComponent, {
        onClick: this.handleOpen
      }, this.props.buttonContent), /*#__PURE__*/react_default.a.createElement(Dialog_default.a, {
        open: this.state.open,
        onClose: this.handleClose
      }, /*#__PURE__*/react_default.a.createElement(DialogTitle_default.a, null, this.props.message), /*#__PURE__*/react_default.a.createElement(DialogActions_default.a, null, /*#__PURE__*/react_default.a.createElement(Button_default.a, {
        onClick: this.handleClose,
        color: "primary"
      }, this.props.abort), /*#__PURE__*/react_default.a.createElement(Button_default.a, {
        onClick: this.handleConfirm,
        color: "primary",
        autoFocus: true
      }, this.props.confirm))));
    }
  }]);
  return ConfirmDialog;
}(react_default.a.Component);
ConfirmDialog_ConfirmDialog.defaultProps = {
  message: 'Are you sure about this?',
  abort: 'No',
  confirm: 'Yes'
};
/* harmony default export */ var components_ConfirmDialog = (ConfirmDialog_ConfirmDialog);
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Table/index.js
var Table = __webpack_require__("../../node_modules/@material-ui/core/Table/index.js");
var Table_default = /*#__PURE__*/__webpack_require__.n(Table);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/TableHead/index.js
var TableHead = __webpack_require__("../../node_modules/@material-ui/core/TableHead/index.js");
var TableHead_default = /*#__PURE__*/__webpack_require__.n(TableHead);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/TableBody/index.js
var TableBody = __webpack_require__("../../node_modules/@material-ui/core/TableBody/index.js");
var TableBody_default = /*#__PURE__*/__webpack_require__.n(TableBody);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/TableRow/index.js
var TableRow = __webpack_require__("../../node_modules/@material-ui/core/TableRow/index.js");
var TableRow_default = /*#__PURE__*/__webpack_require__.n(TableRow);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/TableCell/index.js
var TableCell = __webpack_require__("../../node_modules/@material-ui/core/TableCell/index.js");
var TableCell_default = /*#__PURE__*/__webpack_require__.n(TableCell);

// CONCATENATED MODULE: ./components/Tables.jsx
function Tables_extends() { Tables_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Tables_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var StyledTable = Table_default.a;
var StyledTHead = TableHead_default.a;
var StyledTBody = TableBody_default.a;
var StyledTr = Object(styles["withStyles"])({
  root: {
    '&$selected': {
      backgroundColor: '#b9b9b9'
    }
  },
  selected: {}
})(TableRow_default.a);
var Tables_StyledTh = function StyledTh(props) {
  return /*#__PURE__*/react_default.a.createElement(TableCell_default.a, Tables_extends({
    component: "th"
  }, props));
};
var StyledTd = TableCell_default.a;
var StyledTdKey = StyledTd;
var SysInfoTableContainer = Paper_default.a;
var Tables_SysInfoTable = function SysInfoTable(_ref) {
  var header = _ref.header,
    colspan = _ref.colspan,
    children = _ref.children;
  return /*#__PURE__*/react_default.a.createElement(StyledTable, null, /*#__PURE__*/react_default.a.createElement("thead", null, /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
    colSpan: colspan || 2
  }, header))), /*#__PURE__*/react_default.a.createElement("tbody", null, children));
};
var Tables_SysInfoTableEntry = function SysInfoTableEntry(_ref2) {
  var label = _ref2.label,
    value = _ref2.value,
    values = _ref2.values,
    headers = _ref2.headers;
  if (headers) {
    return /*#__PURE__*/react_default.a.createElement(StyledTr, null, headers.map(function (value) {
      return /*#__PURE__*/react_default.a.createElement(StyledTdKey, null, value || '-');
    }));
  }
  if (values) {
    return /*#__PURE__*/react_default.a.createElement(StyledTr, null, values.map(function (value) {
      return /*#__PURE__*/react_default.a.createElement(StyledTd, null, value || '-');
    }));
  }
  return /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(StyledTdKey, null, label), /*#__PURE__*/react_default.a.createElement(StyledTd, null, value || '-'));
};
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Tabs/index.js
var Tabs = __webpack_require__("../../node_modules/@material-ui/core/Tabs/index.js");
var Tabs_default = /*#__PURE__*/__webpack_require__.n(Tabs);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/Tab/index.js
var Tab = __webpack_require__("../../node_modules/@material-ui/core/Tab/index.js");
var Tab_default = /*#__PURE__*/__webpack_require__.n(Tab);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/IconButton/index.js
var IconButton = __webpack_require__("../../node_modules/@material-ui/core/IconButton/index.js");
var IconButton_default = /*#__PURE__*/__webpack_require__.n(IconButton);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/DialogContent/index.js
var DialogContent = __webpack_require__("../../node_modules/@material-ui/core/DialogContent/index.js");
var DialogContent_default = /*#__PURE__*/__webpack_require__.n(DialogContent);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/vis/dist/vis.js
var vis = __webpack_require__("../../node_modules/vis/dist/vis.js");
var vis_default = /*#__PURE__*/__webpack_require__.n(vis);

// CONCATENATED MODULE: ./modules/GraphView/GraphView.jsx
function GraphView_typeof(obj) { "@babel/helpers - typeof"; return GraphView_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, GraphView_typeof(obj); }
function GraphView_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function GraphView_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function GraphView_createClass(Constructor, protoProps, staticProps) { if (protoProps) GraphView_defineProperties(Constructor.prototype, protoProps); if (staticProps) GraphView_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function GraphView_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) GraphView_setPrototypeOf(subClass, superClass); }
function GraphView_setPrototypeOf(o, p) { GraphView_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return GraphView_setPrototypeOf(o, p); }
function GraphView_createSuper(Derived) { var hasNativeReflectConstruct = GraphView_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = GraphView_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = GraphView_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return GraphView_possibleConstructorReturn(this, result); }; }
function GraphView_possibleConstructorReturn(self, call) { if (call && (GraphView_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return GraphView_assertThisInitialized(self); }
function GraphView_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function GraphView_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function GraphView_getPrototypeOf(o) { GraphView_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return GraphView_getPrototypeOf(o); }




var GraphView_GraphView = /*#__PURE__*/function (_Component) {
  GraphView_inherits(GraphView, _Component);
  var _super = GraphView_createSuper(GraphView);
  function GraphView(props) {
    var _this;
    GraphView_classCallCheck(this, GraphView);
    _this = _super.call(this, props);
    _this.state = {
      nodes: [],
      edges: [],
      network: null
    };
    _this.initNetworkViz = _this.initNetworkViz.bind(GraphView_assertThisInitialized(_this));
    _this.changeNode = _this.changeNode.bind(GraphView_assertThisInitialized(_this));
    return _this;
  }
  GraphView_createClass(GraphView, [{
    key: "prepareData",
    value: function prepareData(nodes, relationships, callback) {
      var nodesArr = [];
      var edgeArr = [];
      nodes.filter(function (node) {
        return !node.skip;
      }).forEach(function (node, index) {
        // var properties = "<ul>";
        // _.forEach(el.properties, function(prop) {
        //     if (!prop.skip) {
        //         if (prop.primaryKey) {
        //             properties += "<li><strong>" + prop.neoKey + "</strong></li>";
        //         } else {
        //             properties += "<li>" + prop.neoKey + "</li>";
        //         }
        //     }
        // });
        // properties += "</ul>";
        var properties = '';
        nodesArr.push({
          id: node.nodeKey,
          label: node.label,
          title: properties,
          color: node.color
        });
      });
      relationships.filter(function (edge) {
        return !edge.skip;
      }).forEach(function (edge, index) {
        var properties = "";
        // var properties = "<ul>";
        // _.forEach(el.properties, function(prop) {
        //     if (!prop.skip) {
        //         properties += "<li>" + prop.neoKey + "</li>";
        //     }
        // });
        // properties += "</ul>";

        var edge = {
          title: properties,
          id: edge.relKey,
          from: edge.start,
          to: edge.end,
          label: edge.type,
          arrows: "to"
        };
        edgeArr.push(edge);
      });
      this.setState({
        nodes: new vis_default.a.DataSet(nodesArr),
        edges: new vis_default.a.DataSet(edgeArr)
      }, callback);
    }
  }, {
    key: "componentDidMount",
    value: function componentDidMount() {
      this.prepareData(this.props.nodes, this.props.relationships, this.initNetworkViz);
    }
  }, {
    key: "initNetworkViz",
    value: function initNetworkViz() {
      var data = {
        nodes: this.state.nodes,
        edges: this.state.edges,
        config: {}
      };
      var visOptions = {};
      var network = new vis_default.a.Network(this.visEl, data, visOptions);
      if (typeof this.props.onNetworkClick === 'function') {
        network.on('click', this.props.onNetworkClick);
      }
      if (typeof this.props.onNetworkDoubleClick === 'function') {
        network.on('doubleClick', this.props.onNetworkDoubleClick);
      }
      this.setState({
        network: network
      });
    }
  }, {
    key: "componentWillReceiveProps",
    value: function componentWillReceiveProps(nextProps) {
      this.prepareData(nextProps.nodes, nextProps.relationships, this.initNetworkViz);
    }
  }, {
    key: "changeNode",
    value: function changeNode(node) {
      this.state.nodes.update({
        id: node.nodeKey,
        label: node.label
      });
    }
  }, {
    key: "changeEdge",
    value: function changeEdge(edge) {
      this.state.edges.update({
        id: edge.relKey,
        label: edge.type
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this2 = this;
      return /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          height: '100%',
          display: 'block'
        },
        ref: function ref(el) {
          _this2.visEl = el;
        }
      });
    }
  }]);
  return GraphView;
}(react["Component"]);
var GraphView_mapStateToProps = function mapStateToProps(state) {
  var mapping = state.mapping;
  var nodeKey = mapping.nodeKey;
  var nodeIndex = mapping.nodeIndex;
  var node = mapping.node;
  var edgeKey = mapping.edgeKey;
  var edgeIndex = mapping.edgeIndex;
  var edge = mapping.edge;
  var value = mapping.value;
  return {
    nodeKey: nodeKey,
    nodeIndex: nodeIndex,
    value: value,
    node: node,
    edgeKey: edgeKey,
    edgeIndex: edgeIndex,
    edge: edge
  };
};
/* harmony default export */ var modules_GraphView_GraphView = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(GraphView_mapStateToProps, null, null, {
  withRef: true
})(GraphView_GraphView)));
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/chroma-js/chroma.js
var chroma = __webpack_require__("../../node_modules/chroma-js/chroma.js");
var chroma_default = /*#__PURE__*/__webpack_require__.n(chroma);

// CONCATENATED MODULE: ./modules/Stream/mapping/ViewMappingSlide.jsx
function ViewMappingSlide_typeof(obj) { "@babel/helpers - typeof"; return ViewMappingSlide_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ViewMappingSlide_typeof(obj); }
function ViewMappingSlide_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function ViewMappingSlide_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ViewMappingSlide_ownKeys(Object(source), !0).forEach(function (key) { ViewMappingSlide_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ViewMappingSlide_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function ViewMappingSlide_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function ViewMappingSlide_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ViewMappingSlide_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ViewMappingSlide_createClass(Constructor, protoProps, staticProps) { if (protoProps) ViewMappingSlide_defineProperties(Constructor.prototype, protoProps); if (staticProps) ViewMappingSlide_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ViewMappingSlide_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ViewMappingSlide_setPrototypeOf(subClass, superClass); }
function ViewMappingSlide_setPrototypeOf(o, p) { ViewMappingSlide_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ViewMappingSlide_setPrototypeOf(o, p); }
function ViewMappingSlide_createSuper(Derived) { var hasNativeReflectConstruct = ViewMappingSlide_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ViewMappingSlide_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ViewMappingSlide_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ViewMappingSlide_possibleConstructorReturn(this, result); }; }
function ViewMappingSlide_possibleConstructorReturn(self, call) { if (call && (ViewMappingSlide_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ViewMappingSlide_assertThisInitialized(self); }
function ViewMappingSlide_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ViewMappingSlide_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ViewMappingSlide_getPrototypeOf(o) { ViewMappingSlide_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ViewMappingSlide_getPrototypeOf(o); }
function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || ViewMappingSlide_unsupportedIterableToArray(arr) || _nonIterableSpread(); }
function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function ViewMappingSlide_unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return ViewMappingSlide_arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return ViewMappingSlide_arrayLikeToArray(o, minLen); }
function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }
function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return ViewMappingSlide_arrayLikeToArray(arr); }
function ViewMappingSlide_arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

















function arrayUpdateAt(array, index, item) {
  return [].concat(_toConsumableArray(array.slice(0, index)), [item], _toConsumableArray(array.slice(index + 1)));
}
function arrayRemoveAt(array, index) {
  return [].concat(_toConsumableArray(array.slice(0, index)), _toConsumableArray(array.slice(index + 1)));
}
var ViewMappingSlide_ViewMappingSlide = /*#__PURE__*/function (_Component) {
  ViewMappingSlide_inherits(ViewMappingSlide, _Component);
  var _super = ViewMappingSlide_createSuper(ViewMappingSlide);
  function ViewMappingSlide(props) {
    var _this;
    ViewMappingSlide_classCallCheck(this, ViewMappingSlide);
    _this = _super.call(this, props);
    _this.state = {
      nodeSelected: -1,
      edgeSelected: -1,
      isMappingSaving: false,
      mappingSaved: false,
      selectedTab: 0,
      highlightedNodes: [],
      highlightedRelationships: [],
      searchNodes: '',
      searchRelationships: '',
      filteredNodes: props.nodes,
      filteredRelationships: props.relationships
    };
    _this.handleChangeTab = _this.handleChangeTab.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.unselectNodeAndEdge = _this.unselectNodeAndEdge.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.changeNode = _this.changeNode.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.onNetworkClick = _this.onNetworkClick.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.onNetworkDoubleClick = _this.onNetworkDoubleClick.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.onNodesSearch = _this.onNodesSearch.bind(ViewMappingSlide_assertThisInitialized(_this));
    _this.onRelationshipsSearch = _this.onRelationshipsSearch.bind(ViewMappingSlide_assertThisInitialized(_this));
    return _this;
  }
  ViewMappingSlide_createClass(ViewMappingSlide, [{
    key: "onNodeClick",
    value: function onNodeClick(index, nodes) {
      this.setState({
        nodeSelected: index,
        edgeSelected: -1
      });
      var myBus = this.props.bus;
      myBus.send(NODE_CLICK, {
        nodeKey: nodes[index].nodeKey,
        nodeIndex: index,
        edgeIndex: -1,
        edgeKey: undefined,
        nodes: this.props.nodes || [],
        relationships: this.props.relationships || [],
        connectionName: this.props.connectionName,
        mappings: this.props.mappings
      });
    }
  }, {
    key: "onEdgeClick",
    value: function onEdgeClick(index, relationships) {
      this.setState({
        edgeSelected: index,
        nodeSelected: -1
      });
      var myBus = this.props.bus;
      myBus.send(EDGE_CLICK, {
        edgeKey: relationships[index].relKey,
        edgeIndex: index,
        nodeIndex: -1,
        nodeKey: undefined,
        nodes: this.props.nodes || [],
        relationships: this.props.relationships || [],
        connectionName: this.props.connectionName,
        mappings: this.props.mappings
      });
    }
  }, {
    key: "unselectNodeAndEdge",
    value: function unselectNodeAndEdge(e) {
      e.preventDefault();
      this.setState({
        nodeSelected: -1,
        edgeSelected: -1
      });
      var myBus = this.props.bus;
      myBus.send(NODE_CLICK, {
        nodeKey: undefined,
        nodeIndex: -1,
        edgeIndex: -1,
        edgeKey: undefined,
        nodes: this.props.nodes || [],
        relationships: this.props.relationships || [],
        connectionName: this.props.connectionName,
        mappings: this.props.mappings
      });
    }
  }, {
    key: "changeLabel",
    value: function changeLabel(node, index, event) {
      node.setLabel(event.target.value);
      this.changeNode(node, index);
      this.graphRef.changeNode(node);
    }
  }, {
    key: "changeNode",
    value: function changeNode(node, index) {
      var updatedNodes = arrayUpdateAt(this.props.nodes, index, node);
      var myBus = this.props.bus;
      myBus.send(LABEL_CHANGED, {
        value: event.target.value,
        node: node,
        nodes: updatedNodes || [],
        relationships: this.props.relationships || [],
        connectionName: this.props.connectionName,
        mappings: this.props.mappings
      });
    }
  }, {
    key: "changeRelType",
    value: function changeRelType(rel, index, event) {
      rel.setType(event.target.value);
      this.changeEdge(rel, index);
      this.graphRef.changeEdge(rel);
    }
  }, {
    key: "changeEdge",
    value: function changeEdge(rel, index) {
      var updatedRelationships = arrayUpdateAt(this.props.relationships, index, rel);
      var myBus = this.props.bus;
      myBus.send(TYPE_CHANGED, {
        value: event.target.value,
        edge: rel,
        nodes: this.props.nodes || [],
        relationships: updatedRelationships || [],
        connectionName: this.props.connectionName,
        mappings: this.props.mappings
      });
    }
  }, {
    key: "changeSelectedNodeColumn",
    value: function changeSelectedNodeColumn(column, index) {
      var nodes = this.props.nodes || [];
      var node = nodes[this.state.nodeSelected] || {};
      var updatedColumns = arrayUpdateAt(node.columns, index, column);
      node.setColumns(updatedColumns);
      this.changeNode(node, this.state.nodeSelected);
    }
  }, {
    key: "changeNodePropertyName",
    value: function changeNodePropertyName(column, index, event) {
      column.setFieldName(event.target.value);
      this.changeSelectedNodeColumn(column, index);
    }
  }, {
    key: "changeNodePropertyNeoType",
    value: function changeNodePropertyNeoType(column, index, event) {
      column.setNeoType(event.target.value);
      this.changeSelectedNodeColumn(column, index);
    }
  }, {
    key: "changeNodePropertySkip",
    value: function changeNodePropertySkip(column, index, event) {
      column.setSkip(event.target.checked);
      this.changeSelectedNodeColumn(column, index);
    }
  }, {
    key: "changeSelectedEdgeColumn",
    value: function changeSelectedEdgeColumn(column, index) {
      var relationships = this.props.relationships || [];
      var edge = relationships[this.state.edgeSelected] || {};
      var updatedColumns = arrayUpdateAt(edge.columns, index, column);
      edge.setColumns(updatedColumns);
      this.changeEdge(edge, this.state.edgeSelected);
    }
  }, {
    key: "changeEdgePropertyName",
    value: function changeEdgePropertyName(column, index, event) {
      column.setFieldName(event.target.value);
      this.changeSelectedEdgeColumn(column, index);
    }
  }, {
    key: "changeEdgePropertyNeoType",
    value: function changeEdgePropertyNeoType(column, index, event) {
      column.setNeoType(event.target.value);
      this.changeSelectedEdgeColumn(column, index);
    }
  }, {
    key: "changeEdgePropertySkip",
    value: function changeEdgePropertySkip(column, index, event) {
      column.setSkip(event.target.checked);
      this.changeSelectedEdgeColumn(column, index);
    }
  }, {
    key: "saveMapping",
    value: function saveMapping() {
      var _this2 = this;
      this.setState({
        isMappingSaving: true
      }, function () {
        if (!_this2.etl) {
          var connection = _this2.props.projectStore.currentProject.getCurrentSourceConnection();
          var importFolder = _this2.props.projectStore.currentProject.getImportFolder() || '';
          _this2.etl = new services_EtlToolService(connection, importFolder);
        }

        // filter skip
        var mappings = _this2.props.mappings.filter(function (item) {
          return !item.skip;
        }).map(function (item) {
          return ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, item), {}, {
            mappings: item.mappings.filter(function (property) {
              return property.skip !== true;
            })
          });
        });
        var currentGraph = _this2.props.projectStore.currentProject.getCurrentGraph();
        _this2.etl.writeMapping(currentGraph, mappings, function (out) {
          console.info(out);
          var msg = showInfoMessage('Mapping saved successfully');
          _this2.props.bus.send(msg.type, msg);
          _this2.setState({
            mappingSaved: true,
            isMappingSaving: false
          });
        }, function (error) {
          console.error(error);
          var actErr = showErrorMessage('Save mapping failed');
          _this2.props.bus.send(actErr.type, actErr);
          _this2.setState({
            isMappingSaving: false
          });
        });
      });
    }
  }, {
    key: "backToStart",
    value: function backToStart(event) {
      event.preventDefault();
      this.props.bus.send(RESTART_WIZARD, {
        restartWizard: true
      });
    }
  }, {
    key: "handleChangeTab",
    value: function handleChangeTab(e, value) {
      this.setState({
        selectedTab: value
      });
    }
  }, {
    key: "skipRelationship",
    value: function skipRelationship(edge, index) {
      var relationshipIndex = _.findIndex(this.props.relationships, {
        relKey: edge.relKey
      });
      var relationships = arrayUpdateAt(this.props.relationships, relationshipIndex, ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, edge), {}, {
        skip: true
      }));
      this.updateFilteredRelationships(relationships);
      var myBus = this.props.bus;
      myBus.send(SKIP_ITEM, {
        nodes: this.props.nodes,
        relationships: relationships,
        mappings: this.props.mappings.map(function (item) {
          return edge.original.name === item.name ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, item), {}, {
            skip: true
          }) : item;
        })
      });
    }
  }, {
    key: "addRelationship",
    value: function addRelationship(edge, index) {
      // Check if nodes are skipped, in which case, we can't use that relationship
      var skippedNode = this.props.nodes.find(function (node) {
        return node.skip && (node.nodeKey === edge.startNode.nodeKey || node.nodeKey === edge.endNode.nodeKey);
      });
      if (skippedNode) {
        var error = showErrorMessage("Cannot add a relationship when one of its nodes is skipped: ".concat(skippedNode.label));
        this.props.bus.send(error.type, error);
      } else {
        var relationshipIndex = _.findIndex(this.props.relationships, {
          relKey: edge.relKey
        });
        var relationships = arrayUpdateAt(this.props.relationships, relationshipIndex, ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, edge), {}, {
          skip: false
        }));
        this.updateFilteredRelationships(relationships);
        var myBus = this.props.bus;
        myBus.send(ADD_ITEM, {
          nodes: this.props.nodes,
          relationships: relationships,
          mappings: this.props.mappings.map(function (item) {
            return edge.original.name === item.name ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, item), {}, {
              skip: false
            }) : item;
          })
        });
      }
    }
  }, {
    key: "addNode",
    value: function addNode(node, index) {
      var nodeIndex = _.findIndex(this.props.nodes, {
        nodeKey: node.nodeKey
      });
      var nodes = arrayUpdateAt(this.props.nodes, nodeIndex, ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, node), {}, {
        skip: false
      }));
      var mappingsToUnskip = [node.original.name];
      var relationships = this.props.relationships.map(function (edge) {
        var using = edge.startNode && edge.startNode.nodeKey === node.nodeKey || edge.endNode && edge.endNode.nodeKey === node.nodeKey;
        if (using) {
          mappingsToUnskip.push(edge.original.name);
        }
        return using ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, edge), {}, {
          skip: false
        }) : edge;
      });
      this.updateFilteredNodes(nodes);
      this.updateFilteredRelationships(relationships);
      var myBus = this.props.bus;
      myBus.send(ADD_ITEM, {
        nodes: nodes,
        relationships: relationships,
        mappings: this.props.mappings.map(function (item) {
          return mappingsToUnskip.includes(item.name) ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, item), {}, {
            skip: false
          }) : item;
        })
      });
    }
  }, {
    key: "skipNode",
    value: function skipNode(node, index) {
      var nodeIndex = _.findIndex(this.props.nodes, {
        nodeKey: node.nodeKey
      });
      var nodes = arrayUpdateAt(this.props.nodes, nodeIndex, ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, node), {}, {
        skip: true
      }));
      var mappingsToSkip = [node.original.name];
      var relationships = this.props.relationships.map(function (edge) {
        var using = edge.startNode && edge.startNode.nodeKey === node.nodeKey || edge.endNode && edge.endNode.nodeKey === node.nodeKey;
        if (using) {
          mappingsToSkip.push(edge.original.name);
        }
        return using ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, edge), {}, {
          skip: true
        }) : edge;
      });
      this.updateFilteredNodes(nodes);
      this.updateFilteredRelationships(relationships);
      var myBus = this.props.bus;
      myBus.send(SKIP_ITEM, {
        nodes: nodes,
        relationships: relationships,
        mappings: this.props.mappings.map(function (item) {
          return mappingsToSkip.includes(item.name) ? ViewMappingSlide_objectSpread(ViewMappingSlide_objectSpread({}, item), {}, {
            skip: true
          }) : item;
        })
      });
    }
  }, {
    key: "onNetworkClick",
    value: function onNetworkClick(params) {
      var nodes = params.nodes,
        items = params.items;
      this.setState({
        highlightedNodes: nodes,
        highlightedRelationships: items
      });
    }
  }, {
    key: "onNetworkDoubleClick",
    value: function onNetworkDoubleClick(params, e) {
      var nodes = params.nodes,
        edges = params.edges;
      if (nodes.length > 0) {
        var nodeIndex = _.findIndex(this.props.nodes, {
          nodeKey: nodes[0]
        });
        if (nodeIndex >= 0) {
          var editEnabled = this.props.nodes[nodeIndex].columns.length > 0;
          if (editEnabled) {
            this.onNodeClick(nodeIndex, this.props.nodes);
          }
        }
      } else if (edges.length > 0) {
        var edgeIndex = _.findIndex(this.props.relationships, {
          relKey: edges[0]
        });
        if (edgeIndex >= 0) {
          var _editEnabled = this.props.relationships[edgeIndex].columns.length > 0;
          if (_editEnabled) {
            this.onEdgeClick(edgeIndex, this.props.relationships);
          }
        }
      }
    }
  }, {
    key: "onNodesSearch",
    value: function onNodesSearch(e) {
      var search = e.target.value;
      this.setState({
        searchNodes: e.target.value
      }, this.updateFilteredNodes);
    }
  }, {
    key: "updateFilteredNodes",
    value: function updateFilteredNodes(nodes) {
      var _this3 = this;
      var currentNodes = typeof nodes === 'undefined' ? this.props.nodes : nodes;
      var filteredNodes = _.filter(currentNodes, function (item) {
        return _.startsWith(_.toLower(item.label), _.toLower(_this3.state.searchNodes));
      });
      this.setState({
        filteredNodes: filteredNodes
      });
    }
  }, {
    key: "onRelationshipsSearch",
    value: function onRelationshipsSearch(e) {
      var search = e.target.value;
      this.setState({
        searchRelationships: e.target.value
      }, this.updateFilteredRelationships);
    }
  }, {
    key: "updateFilteredRelationships",
    value: function updateFilteredRelationships(relationships) {
      var _this4 = this;
      var currentFilteredRelationships = typeof relationships === 'undefined' ? this.props.relationships : relationships;
      var filteredRelationships = _.filter(currentFilteredRelationships, function (item) {
        return _.startsWith(_.toLower(item.type), _.toLower(_this4.state.searchRelationships));
      });
      this.setState({
        filteredRelationships: filteredRelationships
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this5 = this;
      var nodes = this.props.nodes || [];
      var relationships = this.props.relationships || [];
      var node = nodes[this.state.nodeSelected] || {};
      var columns = node && node.columns || [];
      var rel = relationships[this.state.edgeSelected] || {};
      var relColumns = rel && rel.columns || [];
      var nextButtonEnabled = this.state.mappingSaved;
      var selectedTab = this.state.selectedTab;
      return /*#__PURE__*/react_default.a.createElement(styled_StyledFrameSlide, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameTitleBar, null, /*#__PURE__*/react_default.a.createElement(Headers_H3, null, "Explore and change your metadata"))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(SysInfoTableContainer, null, /*#__PURE__*/react_default.a.createElement(Tabs_default.a, {
        value: selectedTab,
        onChange: this.handleChangeTab
      }, /*#__PURE__*/react_default.a.createElement(Tab_default.a, {
        label: "Nodes"
      }), /*#__PURE__*/react_default.a.createElement(Tab_default.a, {
        label: "Relationships"
      })), selectedTab === 0 && /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          maxHeight: '400px',
          overflow: 'auto'
        }
      }, /*#__PURE__*/react_default.a.createElement(StyledTable, null, /*#__PURE__*/react_default.a.createElement(StyledTHead, null, /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        colSpan: 3
      }, /*#__PURE__*/react_default.a.createElement(TextInput, {
        placeholder: "Search by entity name",
        value: this.state.searchNodes,
        onChange: this.onNodesSearch,
        fullWidth: true
      }))), /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '68px',
          textAlign: 'center',
          paddingRight: '24px'
        }
      }), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Entity Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '50px',
          textAlign: 'center',
          paddingRight: '24px'
        }
      }, "Actions"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '50px',
          textAlign: 'center'
        }
      }, "Skip"))), /*#__PURE__*/react_default.a.createElement(StyledTBody, null, this.state.filteredNodes.map(function (node, index) {
        var editEnabled = node.columns.length > 0;
        var skipped = node.skip;
        return /*#__PURE__*/react_default.a.createElement(StyledTr, {
          key: index,
          selected: _this5.state.highlightedNodes.indexOf(node.nodeKey) >= 0
        }, /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            paddingRight: '24px'
          }
        }, /*#__PURE__*/react_default.a.createElement(DotColor, {
          color: node.color
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(TextInput, {
          value: node.label,
          onChange: _this5.changeLabel.bind(_this5, node, index),
          onClick: function onClick(e) {
            e.stopPropagation();
          },
          fullWidth: true
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            textAlign: 'right',
            paddingRight: '24px'
          }
        }, editEnabled && /*#__PURE__*/react_default.a.createElement(IconButton_default.a, {
          title: "Edit Node",
          onClick: _this5.onNodeClick.bind(_this5, index, _this5.state.filteredNodes)
        }, /*#__PURE__*/react_default.a.createElement(Icon_default.a, null, "edit"))), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(Checkbox_default.a, {
          checked: skipped,
          onChange: function onChange(event, checked) {
            checked ? _this5.skipNode(node, index) : _this5.addNode(node, index);
          }
        })));
      })))), selectedTab === 1 && /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          maxHeight: '400px',
          overflow: 'auto'
        }
      }, /*#__PURE__*/react_default.a.createElement(StyledTable, null, /*#__PURE__*/react_default.a.createElement(StyledTHead, null, /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        colSpan: 3
      }, /*#__PURE__*/react_default.a.createElement(TextInput, {
        placeholder: "Search by entity name",
        value: this.state.searchRelationships,
        onChange: this.onRelationshipsSearch,
        fullWidth: true
      }))), /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Entity Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '150px',
          paddingRight: '24px'
        }
      }, "Start Node"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '150px',
          paddingRight: '24px'
        }
      }, "End Node"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '50px',
          paddingRight: '24px',
          textAlign: 'center'
        }
      }, "Actions"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          width: '50px',
          textAlign: 'center'
        }
      }, "Skip"))), /*#__PURE__*/react_default.a.createElement(StyledTBody, null, this.state.filteredRelationships.map(function (item, index) {
        var editEnabled = item.columns.length > 0;
        var skipped = item.skip;
        return /*#__PURE__*/react_default.a.createElement(StyledTr, {
          key: index,
          selected: _.findIndex(_this5.state.highlightedRelationships, {
            edgeId: item.relKey
          }) >= 0
        }, /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            minWidth: '200px',
            paddingRight: '24px'
          }
        }, /*#__PURE__*/react_default.a.createElement(TextInput, {
          value: item.type,
          onChange: _this5.changeRelType.bind(_this5, item, index),
          onClick: function onClick(e) {
            e.stopPropagation();
          },
          fullWidth: true
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            minWidth: '100px',
            paddingRight: '24px',
            wordBreak: 'break-word'
          }
        }, item.startNode && item.startNode.label || ''), /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            minWidth: '100px',
            paddingRight: '24px',
            wordBreak: 'break-word'
          }
        }, item.endNode && item.endNode.label || ''), /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            textAlign: 'right',
            paddingRight: '24px'
          }
        }, editEnabled && /*#__PURE__*/react_default.a.createElement(IconButton_default.a, {
          title: "Edit Relationship",
          onClick: _this5.onEdgeClick.bind(_this5, index, _this5.state.filteredRelationships)
        }, /*#__PURE__*/react_default.a.createElement(Icon_default.a, null, "edit"))), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(Checkbox_default.a, {
          checked: skipped,
          onChange: function onChange(event, checked) {
            checked ? _this5.skipRelationship(item, index) : _this5.addRelationship(item, index);
          }
        })));
      })))))), /*#__PURE__*/react_default.a.createElement(Dialog_default.a, {
        open: this.state.nodeSelected !== -1,
        onClose: this.unselectNodeAndEdge,
        maxWidth: "md"
      }, /*#__PURE__*/react_default.a.createElement(DialogContent_default.a, null, /*#__PURE__*/react_default.a.createElement(StyledTable, null, /*#__PURE__*/react_default.a.createElement(StyledTHead, null, /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Column Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Property Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "SQL Type"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Neo4j Type"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Primary Key"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          textAlign: 'center'
        }
      }, "Skip"))), /*#__PURE__*/react_default.a.createElement(StyledTBody, null, console.log(columns), columns.map(function (item, index) {
        return /*#__PURE__*/react_default.a.createElement(StyledTr, {
          key: index
        }, /*#__PURE__*/react_default.a.createElement(StyledTd, null, item.columnName), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(TextInput, {
          style: {
            cursor: item.columnName == '_ROW_INDEX_' ? 'not-allowed' : 'default'
          },
          value: item.fieldName,
          disabled: item.columnName == '_ROW_INDEX_',
          onChange: _this5.changeNodePropertyName.bind(_this5, item, index)
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, null, item.sqlType), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(StyledSelect, {
          style: {
            cursor: item.columnName == '_ROW_INDEX_' ? 'not-allowed' : 'default'
          },
          value: item.neo4jType,
          disabled: item.columnName == '_ROW_INDEX_',
          onChange: _this5.changeNodePropertyNeoType.bind(_this5, item, index)
        }, /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Double"
        }, "Double"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "String"
        }, "String"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Long"
        }, "Long"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Byte"
        }, "Byte"))), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(Form_CheckboxSelector, {
          checked: item.primaryKey,
          readOnly: true,
          disabled: true
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(Form_CheckboxSelector, {
          checked: item.skip,
          onChange: _this5.changeNodePropertySkip.bind(_this5, item, index)
        })));
      })))), /*#__PURE__*/react_default.a.createElement(DialogActions_default.a, null, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.unselectNodeAndEdge
      }, "Save"))), /*#__PURE__*/react_default.a.createElement(Dialog_default.a, {
        open: this.state.edgeSelected !== -1,
        onClose: this.unselectNodeAndEdge,
        maxWidth: "md"
      }, /*#__PURE__*/react_default.a.createElement(DialogContent_default.a, null, /*#__PURE__*/react_default.a.createElement(StyledTable, null, /*#__PURE__*/react_default.a.createElement(StyledTHead, null, /*#__PURE__*/react_default.a.createElement(StyledTr, null, /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Column Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Property Name"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "SQL Type"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, null, "Neo4j Type"), /*#__PURE__*/react_default.a.createElement(Tables_StyledTh, {
        style: {
          textAlign: 'center'
        }
      }, "Skip"))), /*#__PURE__*/react_default.a.createElement(StyledTBody, null, relColumns.map(function (item, index) {
        return /*#__PURE__*/react_default.a.createElement(StyledTr, {
          key: index
        }, /*#__PURE__*/react_default.a.createElement(StyledTd, null, item.columnName), /*#__PURE__*/react_default.a.createElement(StyledTd, {
          style: {
            padding: '2px'
          }
        }, /*#__PURE__*/react_default.a.createElement(TextInput, {
          value: item.fieldName,
          onChange: _this5.changeEdgePropertyName.bind(_this5, item, index),
          fullWidth: true
        })), /*#__PURE__*/react_default.a.createElement(StyledTd, null, item.sqlType), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(StyledSelect, {
          value: item.neo4jType,
          onChange: _this5.changeEdgePropertyNeoType.bind(_this5, item, index)
        }, /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Double"
        }, "Double"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "String"
        }, "String"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Long"
        }, "Long"), /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          value: "Byte"
        }, "Byte"))), /*#__PURE__*/react_default.a.createElement(StyledTd, null, /*#__PURE__*/react_default.a.createElement(Form_CheckboxSelector, {
          checked: item.skip,
          onChange: _this5.changeEdgePropertySkip.bind(_this5, item, index)
        })));
      })))), /*#__PURE__*/react_default.a.createElement(DialogActions_default.a, null, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.unselectNodeAndEdge
      }, "Save"))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6,
        style: {
          height: '472px'
        }
      }, /*#__PURE__*/react_default.a.createElement(modules_GraphView_GraphView, {
        nodes: this.props.nodes,
        relationships: this.props.relationships,
        ref: function ref(r) {
          r ? _this5.graphRef = r.getWrappedInstance() : null;
        },
        onNetworkClick: this.onNetworkClick,
        onNetworkDoubleClick: this.onNetworkDoubleClick
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(Divider_default.a, null)), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.backToStart.bind(this),
        label: "Back to Start",
        disabled: this.props.isMappingSaving
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6,
        style: {
          textAlign: 'right'
        }
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        style: {
          marginRight: '8px'
        },
        onClick: this.saveMapping.bind(this),
        label: "Save Mapping",
        isLoading: this.props.isMappingSaving
      }), /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.props.next,
        disabled: !nextButtonEnabled,
        label: "Next"
      })));
    }
  }]);
  return ViewMappingSlide;
}(react["Component"]);
var ViewMappingSlide_mapStateToProps = function mapStateToProps(state) {
  var mapping = state.mapping;
  return {
    nodes: mapping.nodes,
    relationships: mapping.relationships,
    connectionName: mapping.connectionName,
    mappings: mapping.mappings,
    projectStore: state.projectStore
  };
};
/* harmony default export */ var mapping_ViewMappingSlide = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(ViewMappingSlide_mapStateToProps, null)(ViewMappingSlide_ViewMappingSlide)));
// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/FormControl/index.js
var FormControl = __webpack_require__("../../node_modules/@material-ui/core/FormControl/index.js");
var FormControl_default = /*#__PURE__*/__webpack_require__.n(FormControl);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/InputLabel/index.js
var InputLabel = __webpack_require__("../../node_modules/@material-ui/core/InputLabel/index.js");
var InputLabel_default = /*#__PURE__*/__webpack_require__.n(InputLabel);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/FormHelperText/index.js
var FormHelperText = __webpack_require__("../../node_modules/@material-ui/core/FormHelperText/index.js");
var FormHelperText_default = /*#__PURE__*/__webpack_require__.n(FormHelperText);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/TextField/index.js
var TextField = __webpack_require__("../../node_modules/@material-ui/core/TextField/index.js");
var TextField_default = /*#__PURE__*/__webpack_require__.n(TextField);

// EXTERNAL MODULE: /Users/mh/d/java/neo/neo4j-etl/neo4j-etl-ui/node_modules/@material-ui/core/FormControlLabel/index.js
var FormControlLabel = __webpack_require__("../../node_modules/@material-ui/core/FormControlLabel/index.js");
var FormControlLabel_default = /*#__PURE__*/__webpack_require__.n(FormControlLabel);

// CONCATENATED MODULE: ./modules/Stream/mapping/ImportDataSlide.jsx
function ImportDataSlide_typeof(obj) { "@babel/helpers - typeof"; return ImportDataSlide_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ImportDataSlide_typeof(obj); }
function ImportDataSlide_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function ImportDataSlide_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ImportDataSlide_ownKeys(Object(source), !0).forEach(function (key) { ImportDataSlide_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ImportDataSlide_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function ImportDataSlide_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function ImportDataSlide_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ImportDataSlide_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ImportDataSlide_createClass(Constructor, protoProps, staticProps) { if (protoProps) ImportDataSlide_defineProperties(Constructor.prototype, protoProps); if (staticProps) ImportDataSlide_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ImportDataSlide_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ImportDataSlide_setPrototypeOf(subClass, superClass); }
function ImportDataSlide_setPrototypeOf(o, p) { ImportDataSlide_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ImportDataSlide_setPrototypeOf(o, p); }
function ImportDataSlide_createSuper(Derived) { var hasNativeReflectConstruct = ImportDataSlide_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ImportDataSlide_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ImportDataSlide_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ImportDataSlide_possibleConstructorReturn(this, result); }; }
function ImportDataSlide_possibleConstructorReturn(self, call) { if (call && (ImportDataSlide_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ImportDataSlide_assertThisInitialized(self); }
function ImportDataSlide_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ImportDataSlide_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ImportDataSlide_getPrototypeOf(o) { ImportDataSlide_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ImportDataSlide_getPrototypeOf(o); }














var ImportDataSlide_ImportDataSlide = /*#__PURE__*/function (_Component) {
  ImportDataSlide_inherits(ImportDataSlide, _Component);
  var _super = ImportDataSlide_createSuper(ImportDataSlide);
  function ImportDataSlide(props) {
    var _this;
    ImportDataSlide_classCallCheck(this, ImportDataSlide);
    _this = _super.call(this, props);
    _this.onGraphClick = _this.onGraphClick.bind(ImportDataSlide_assertThisInitialized(_this));
    _this.toggleLogs = _this.toggleLogs.bind(ImportDataSlide_assertThisInitialized(_this));
    _this.state = {
      output: '',
      waitingMessage: '',
      multiline: true,
      multilineOptions: {
        'multiline-fields': '' + true
      },
      showLogs: false,
      importModes: [],
      currentMode: '',
      unwindBatchSize: DEFAULT_UNWIND_BATCH_SIZE,
      txBatchSize: DEFAULT_TX_BATCH_SIZE,
      selectedGraph: props.projectStore.currentProject.getCurrentGraph(),
      databaseName: '',
      isImporting: false,
      dbAutoCreation: false,
      databaseNameOptions: [/*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
        value: ""
      }, "Default Database")]
    };
    return _this;
  }
  ImportDataSlide_createClass(ImportDataSlide, [{
    key: "componentDidMount",
    value: function componentDidMount() {
      this.setSelectedGraph(this.props.projectStore.currentProject.getCurrentGraph());
    }
  }, {
    key: "componentWillReceiveProps",
    value: function componentWillReceiveProps(nextProps) {
      this.setSelectedGraph(nextProps.projectStore.currentProject.getCurrentGraph());
    }
  }, {
    key: "setSelectedGraph",
    value: function setSelectedGraph(selectedGraph) {
      if (this.state.isImporting) {
        return null;
      }
      var currentMode = '';
      var importModes = IMPORT_MODES.map(function (item, index) {
        var databaseStatus = selectedGraph.status;
        var disabled = databaseStatus === 'ACTIVE' && item.mode === 'offline' || databaseStatus === 'INACTIVE' && item.mode === 'online';
        if (!currentMode && !disabled) {
          currentMode = item.cmd;
        }
        return {
          value: item.cmd,
          key: index,
          disabled: disabled,
          children: item.name,
          mode: item.mode
        };
      });
      if (selectedGraph.status === 'ACTIVE') {
        this.databaseList();
      }
      this.setState({
        importModes: importModes,
        currentMode: currentMode,
        selectedGraph: selectedGraph
      });
    }
  }, {
    key: "runImport",
    value: function runImport(currentGraph) {
      var _this2 = this;
      var cmd = "COMMAND: " + this.etl.buildShellCommandOfImport(currentGraph);
      var outMsg = cmd + "\nLOG:\n";
      console.info(outMsg);
      this.setState({
        output: outMsg,
        waitingMessage: '',
        isImporting: true
      }, function () {
        var processListener = new services_ProcessListener();
        processListener.setErrorListener(function (out) {
          var outAppend = _this2.state.output + out;
          _this2.setState({
            output: outAppend,
            waitingMessage: ''
          });
          var message = out.message || out;
          if (message.toUpperCase().indexOf('java is restricted to execute'.toUpperCase()) !== -1) {
            myBus.send(JAVA_PERMISSION_ERROR_SHOW);
          }
        });
        var start = new Date().getTime();
        processListener.setSuccessHandler(function (out) {
          var metricsFromDB = JSON.parse(out.stdout || '{}');
          if (Object.keys(metricsFromDB).length) {
            console.log(currentGraph);
            var extra = {
              'import-mode': _this2.state.currentMode,
              'database-version': currentGraph.connection.info.version,
              'database-edition': currentGraph.connection.info.edition,
              'database-type': currentGraph.connection.type,
              elapsed: new Date().getTime() - start
            };
            var importMetrics = ImportDataSlide_objectSpread(ImportDataSlide_objectSpread({}, extra), metricsFromDB);
            console.log('Sending %o to metrics', importMetrics);
            window.neo4jDesktopApi.sendMetrics('neo4j-etl', 'import', importMetrics);
          }
          var msg = showInfoMessage('Import successful');
          _this2.props.bus.send(msg.type, msg);
          _this2.setState({
            isImporting: false
          });
        });
        processListener.setFailureHandler(function (error) {
          console.error(error);
          _this2.setState({
            waitingMessage: '',
            output: cmd + "\n" + error,
            isImporting: false
          });
          var msg = showErrorMessage('Import failed see the logs below for further details');
          _this2.props.bus.send(msg.type, msg);
        });
        _this2.etl.importData(currentGraph, _this2.state.multilineOptions, processListener);
      });
    }
  }, {
    key: "initEtl",
    value: function initEtl() {
      if (!this.etl) {
        var connection = this.props.projectStore.currentProject.getCurrentSourceConnection();
        var importFolder = this.props.projectStore.currentProject.getImportFolder() || '';
        this.etl = new services_EtlToolService(connection, importFolder);
      }
    }
  }, {
    key: "importData",
    value: function importData() {
      this.initEtl();
      var currentMode = this.state.currentMode;
      this.etl.setImportMode(currentMode);
      var isOnline = this.etl.isOnlineMode(currentMode);
      var myBus = this.props.bus;
      var currentProject = this.props.projectStore.currentProject;
      var neo4j = currentProject.getCurrentGraph();
      if (!neo4j) {
        var msg = showErrorMessage('Select a valid Neo4j instance before running the import');
        myBus.send(msg.type, msg);
        return;
      }
      var databaseStatus = neo4j.status;
      var connectionInfo = neo4j.connection.info;
      var dbConnection = currentProject.getCurrentSourceConnection();
      this.etl.setDatabaseName(this.state.databaseName);
      if (!isOnline) {
        this.etl.setDbAutoCreation(this.state.dbAutoCreation);
      }

      // Run offline import only if database is shut down
      if (!isOnline) {
        if (databaseStatus === 'INACTIVE') {
          var _msg2 = showInfoMessage(null);
          myBus.send(_msg2.type, _msg2);
          this.setState({
            waitingMessage: "Importing data from ".concat(dbConnection.name, " to ").concat(neo4j.name, " (").concat(connectionInfo.version, " ").concat(connectionInfo.edition, ") waiting for loading ...")
          });
          this.runImport(neo4j);
          return;
        }
        var _msg = showErrorMessage(currentMode + ' must be run only if the database is offline');
        myBus.send(_msg.type, _msg);
      }
      if (isOnline) {
        if (currentMode === 'cypher:batch' || currentMode === 'cypher:fromSQL') {
          var unwindBatchSize = this.state.unwindBatchSize;
          var txBatchSize = this.state.txBatchSize;
          if (unwindBatchSize > 0 && txBatchSize > 0 && unwindBatchSize > txBatchSize) {
            var _msg4 = showErrorMessage('Unwind import mode can be run only if Unwind Row Size is less or equal then Transaction Batch Size');
            myBus.send(_msg4.type, _msg4);
            return;
          } else if (!unwindBatchSize || !txBatchSize || unwindBatchSize && unwindBatchSize === 0 || txBatchSize && txBatchSize === 0) {
            var _msg5 = showErrorMessage('Unwind Row Size and Transaction Batch Size must be defined with Unwind import mode');
            myBus.send(_msg5.type, _msg5);
            return;
          } else {
            this.etl.setUnwindBatchSize(unwindBatchSize);
            this.etl.setTxBatchSize(txBatchSize);
          }
        }
        if (databaseStatus === 'ACTIVE') {
          var _msg6 = showInfoMessage(null);
          myBus.send(_msg6.type, _msg6);
          this.setState({
            waitingMessage: "Importing data from ".concat(dbConnection.name, " to ").concat(neo4j.name, " (").concat(connectionInfo.version, " ").concat(connectionInfo.edition, ") waiting for loading ...")
          });
          this.runImport(neo4j);
          return;
        }
        var _msg3 = showErrorMessage(currentMode + ' must be run only if the database is online');
        myBus.send(_msg3.type, _msg3);
      }
    }
  }, {
    key: "changeImportCommand",
    value: function changeImportCommand(event) {
      console.info("changing import command", event.target.value);
      this.initEtl();
      this.setState({
        currentMode: event.target.value
      });
    }
  }, {
    key: "changeUnwindBatchSize",
    value: function changeUnwindBatchSize(event) {
      this.setState({
        unwindBatchSize: event.target.value
      });
    }
  }, {
    key: "changeTxBatchSize",
    value: function changeTxBatchSize(event) {
      this.setState({
        txBatchSize: event.target.value
      });
    }
  }, {
    key: "changeDatabaseName",
    value: function changeDatabaseName(event) {
      this.setState({
        databaseName: event.target.value
      });
    }
  }, {
    key: "changeDbAutoCreation",
    value: function changeDbAutoCreation(event, checked) {
      this.setState({
        dbAutoCreation: checked
      });
    }
  }, {
    key: "backToStart",
    value: function backToStart(event) {
      event.preventDefault();
      this.props.bus.send(RESTART_WIZARD, {
        restartWizard: true
      });
    }
  }, {
    key: "toggleLogs",
    value: function toggleLogs() {
      this.setState(function (state) {
        return {
          showLogs: !state.showLogs
        };
      });
    }
  }, {
    key: "onGraphClick",
    value: function onGraphClick(graph) {
      this.props.projectStore.currentProject.setCurrentGraph(graph);
      this.setSelectedGraph(graph);
    }
  }, {
    key: "databaseList",
    value: function databaseList() {
      var _this3 = this;
      this.initEtl();
      var convertArrayToMenuItem = function convertArrayToMenuItem(list) {
        var opts = {
          databaseNameOptions: list.map(function (elem) {
            return {
              value: elem || '',
              children: elem || 'Default Database'
            };
          }).map(function (elem) {
            return /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, elem);
          }),
          databaseName: list[0] || ''
        };
        _this3.setState(function (state) {
          return opts;
        });
      };
      this.etl.getDatabaseList(this.state.selectedGraph, convertArrayToMenuItem, function () {
        return convertArrayToMenuItem(['']);
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this4 = this;
      var selectedGraph = this.state.selectedGraph || {};
      var dbConnection = this.props.projectStore.currentProject.getCurrentSourceConnection();
      var graphs = this.props.projectStore.currentProject.getAllGraphs();
      var nextButtonEnabled = selectedGraph && this.state.currentMode;
      return /*#__PURE__*/react_default.a.createElement(styled_StyledFrameSlide, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameTitleBar, null, /*#__PURE__*/react_default.a.createElement(Headers_H3, null, "Import your data into Neo4j"))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(StyledFrameMainSection, null, /*#__PURE__*/react_default.a.createElement(Text_Lead, {
        component: "div"
      }, "Press the button and start import your data"))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 4
      }, /*#__PURE__*/react_default.a.createElement(Card_default.a, null, /*#__PURE__*/react_default.a.createElement(CardContent_default.a, null, /*#__PURE__*/react_default.a.createElement("strong", null, "FROM"), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "Connection name: ", /*#__PURE__*/react_default.a.createElement("i", null, dbConnection.name)), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "From database: ", /*#__PURE__*/react_default.a.createElement("i", null, dbConnection && dbConnection.database.toUpperCase())), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "Of type: ", /*#__PURE__*/react_default.a.createElement("i", null, dbConnection && dbConnection.driverType.toUpperCase()))))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 4,
        style: {
          textAlign: 'center'
        }
      }, /*#__PURE__*/react_default.a.createElement("div", null, /*#__PURE__*/react_default.a.createElement(Icon_default.a, {
        style: {
          fontSize: '100px'
        }
      }, "trending_flat")), /*#__PURE__*/react_default.a.createElement(FormControl_default.a, {
        style: {
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(InputLabel_default.a, {
        shrink: true,
        htmlFor: "importMode"
      }, /*#__PURE__*/react_default.a.createElement("b", null, "Import Mode")), /*#__PURE__*/react_default.a.createElement(StyledSelect, {
        id: "importMode",
        style: {
          width: '100%'
        },
        onChange: this.changeImportCommand.bind(this),
        value: this.state.currentMode,
        disabled: this.state.isImporting
      }, "required=", true, this.state.importModes.filter(function (item) {
        return item.value !== 'cypher:direct';
      }) // @todo we exclude the LOAD CSV import for now
      .filter(function (item) {
        /*
          Adds only the correct import mode based on the selected graph connection type and status.
          1) if chosen graph is an active remote graph instance, then the only available import mode will be 'Online import batch'
          2) if chosen graph is an active local graph instance, then the import modes available will be 'Online import Load csv' or 'Online import batch'
          3) if chosen graph is an inactive local graph instance, then the only available import mode will be 'Bulk Import'
        */

        if (_this4.state.selectedGraph.connection.type === 'REMOTE') {
          return _this4.state.selectedGraph.status === 'ACTIVE' && (item.value === 'cypher:batch' || item.value === 'cypher:fromSQL');
        }
        if (_this4.state.selectedGraph.connection.type === 'LOCAL') {
          return _this4.state.selectedGraph.status === 'ACTIVE' && item.mode === 'online' || _this4.state.selectedGraph.status === 'INACTIVE' && item.mode === 'offline';
        }
        return false;
      }).map(function (item) {
        return /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, item);
      })), /*#__PURE__*/react_default.a.createElement(FormHelperText_default.a, null, "The import mode type")), /*#__PURE__*/react_default.a.createElement("br", null), /*#__PURE__*/react_default.a.createElement("br", null), (this.state.currentMode === 'cypher:batch' || this.state.currentMode === 'cypher:fromSQL' || this.state.currentMode === 'cypher:direct') && /*#__PURE__*/react_default.a.createElement(FormControl_default.a, {
        style: {
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(InputLabel_default.a, {
        shrink: true,
        htmlFor: "databaseName"
      }, /*#__PURE__*/react_default.a.createElement("b", null, "Neo4j Database")), /*#__PURE__*/react_default.a.createElement(StyledSelect, {
        id: "databaseName",
        displayEmpty: true,
        style: {
          width: '100%'
        },
        value: this.state.databaseName,
        onChange: this.changeDatabaseName.bind(this),
        required: this.state.databaseNameOptions.length > 0 && (this.state.currentMode === 'cypher:batch' || this.state.currentMode === 'cypher:fromSQL'),
        disabled: this.state.isImporting
      }, this.state.databaseNameOptions), /*#__PURE__*/react_default.a.createElement(FormHelperText_default.a, null, "Select the target database")), (this.state.currentMode === 'cypher:batch' || this.state.currentMode === 'cypher:fromSQL') && /*#__PURE__*/react_default.a.createElement("div", null, /*#__PURE__*/react_default.a.createElement(FormControl_default.a, {
        style: {
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        id: "unwindBatchSize",
        label: /*#__PURE__*/react_default.a.createElement("b", null, "Unwind Row Size"),
        value: this.state.unwindBatchSize,
        defaultValue: DEFAULT_UNWIND_BATCH_SIZE,
        description: 'Default value is ' + DEFAULT_UNWIND_BATCH_SIZE,
        type: "number",
        onChange: this.changeUnwindBatchSize.bind(this),
        margin: "normal",
        required: this.state.currentMode === 'cypher:batch' || this.state.currentMode === 'cypher:fromSQL',
        disabled: this.state.isImporting,
        style: {
          width: '100%'
        }
      }), /*#__PURE__*/react_default.a.createElement(FormHelperText_default.a, null, "Number of rows processed per UNWIND")), /*#__PURE__*/react_default.a.createElement(FormControl_default.a, {
        style: {
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        id: "txBatchSize",
        label: /*#__PURE__*/react_default.a.createElement("b", null, "Transaction Batch Size"),
        value: this.state.txBatchSize,
        defaultValue: DEFAULT_TX_BATCH_SIZE,
        description: 'Default value is ' + DEFAULT_TX_BATCH_SIZE,
        type: "number",
        onChange: this.changeTxBatchSize.bind(this),
        margin: "normal",
        required: this.state.currentMode === 'cypher:batch' || this.state.currentMode === 'cypher:fromSQL',
        disabled: this.state.isImporting,
        style: {
          width: '100%'
        }
      }), /*#__PURE__*/react_default.a.createElement(FormHelperText_default.a, null, "Number of rows processed per Transaction"))), this.state.currentMode === 'bulk:neo4j-import' && /*#__PURE__*/react_default.a.createElement("div", null, /*#__PURE__*/react_default.a.createElement(FormControl_default.a, {
        style: {
          width: '100%'
        }
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        id: "databaseNamePlain",
        label: /*#__PURE__*/react_default.a.createElement("b", null, "Neo4j database name"),
        description: "Neo4j database name (if empty we use the default db)",
        value: this.state.databaseName,
        onChange: this.changeDatabaseName.bind(this),
        margin: "normal",
        disabled: this.state.isImporting,
        style: {
          width: '100%'
        },
        inputProps: {
          'min-length': 3
        }
      }), /*#__PURE__*/react_default.a.createElement(FormHelperText_default.a, null, "(Only for Neo4j version >= 4.0) If not specified we use the default db")), /*#__PURE__*/react_default.a.createElement(FormControlLabel_default.a, {
        control: /*#__PURE__*/react_default.a.createElement(Checkbox_default.a, {
          id: "dbAutoCreation",
          checked: this.state.dbAutoCreation,
          onChange: this.changeDbAutoCreation.bind(this),
          value: "dbAutoCreation"
        }),
        label: /*#__PURE__*/react_default.a.createElement("b", null, "Auto create the Database")
      }))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 4
      }, /*#__PURE__*/react_default.a.createElement(Card_default.a, null, /*#__PURE__*/react_default.a.createElement(CardContent_default.a, null, /*#__PURE__*/react_default.a.createElement("strong", null, "TO"), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "Instance name: ", /*#__PURE__*/react_default.a.createElement("i", null, selectedGraph.name)), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "Version: ", /*#__PURE__*/react_default.a.createElement("i", null, selectedGraph.connection.info.version, " - ", selectedGraph.connection.info.edition, " (", selectedGraph.connection.type, ")")), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          paddingLeft: '20px',
          paddingTop: '10px'
        }
      }, "Status: ", /*#__PURE__*/react_default.a.createElement("i", null, selectedGraph.connection.databaseStatus))))), this.state.showLogs && /*#__PURE__*/react_default.a.createElement(react_default.a.Fragment, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(Divider_default.a, null)), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12,
        style: {
          height: '200px',
          overflowY: 'scroll'
        }
      }, /*#__PURE__*/react_default.a.createElement("pre", null, this.state.output), /*#__PURE__*/react_default.a.createElement("div", {
        style: {
          textAlign: 'center',
          fontSize: '18px',
          marginTop: '20px'
        }
      }, this.state.waitingMessage))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(Divider_default.a, null)), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.backToStart.bind(this),
        label: "Back to Start",
        disabled: this.state.isImporting
      }), " ", /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.props.prev,
        label: "Back to Mapping",
        disabled: this.state.isImporting
      }), " ", this.state.output && /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.toggleLogs,
        label: "".concat(this.state.showLogs ? 'Hide' : 'See', " Logs")
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6,
        style: {
          textAlign: 'right'
        }
      }, /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.importData.bind(this),
        label: "Import Data",
        isLoading: this.state.isImporting
      })));
    }
  }]);
  return ImportDataSlide;
}(react["Component"]);
var ImportDataSlide_mapStateToProps = function mapStateToProps(state) {
  var mapping = state.mapping;
  return {
    connectionName: mapping.connectionName,
    mappings: mapping.mappings,
    projectStore: state.projectStore
  };
};
/* harmony default export */ var mapping_ImportDataSlide = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(ImportDataSlide_mapStateToProps, null)(ImportDataSlide_ImportDataSlide)));
// CONCATENATED MODULE: ./modules/Stream/mapping/CarouselMapping.jsx
function CarouselMapping_typeof(obj) { "@babel/helpers - typeof"; return CarouselMapping_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, CarouselMapping_typeof(obj); }
function CarouselMapping_extends() { CarouselMapping_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return CarouselMapping_extends.apply(this, arguments); }
function CarouselMapping_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function CarouselMapping_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function CarouselMapping_createClass(Constructor, protoProps, staticProps) { if (protoProps) CarouselMapping_defineProperties(Constructor.prototype, protoProps); if (staticProps) CarouselMapping_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function CarouselMapping_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) CarouselMapping_setPrototypeOf(subClass, superClass); }
function CarouselMapping_setPrototypeOf(o, p) { CarouselMapping_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return CarouselMapping_setPrototypeOf(o, p); }
function CarouselMapping_createSuper(Derived) { var hasNativeReflectConstruct = CarouselMapping_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = CarouselMapping_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = CarouselMapping_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return CarouselMapping_possibleConstructorReturn(this, result); }; }
function CarouselMapping_possibleConstructorReturn(self, call) { if (call && (CarouselMapping_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return CarouselMapping_assertThisInitialized(self); }
function CarouselMapping_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function CarouselMapping_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function CarouselMapping_getPrototypeOf(o) { CarouselMapping_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return CarouselMapping_getPrototypeOf(o); }












var CarouselMapping_CarouselMapping = /*#__PURE__*/function (_Component) {
  CarouselMapping_inherits(CarouselMapping, _Component);
  var _super = CarouselMapping_createSuper(CarouselMapping);
  function CarouselMapping(props) {
    var _this;
    CarouselMapping_classCallCheck(this, CarouselMapping);
    _this = _super.call(this, props);
    _this.slides = _this.props.slides || [mapping_StartMappingSlide, mapping_ViewMappingSlide, mapping_ImportDataSlide];
    _this.state = {
      visibleSlide: 0,
      connectionName: '',
      connectionStorage: new services_ConnectionStorage(),
      waitingMessage: '',
      colors: chroma_default.a.scale(['#ee6445', '#ffffd9']).colors(15),
      selectedColor: '#b9b9b9'
    };
    return _this;
  }
  CarouselMapping_createClass(CarouselMapping, [{
    key: "componentWillReceiveProps",
    value: function componentWillReceiveProps(nextProps) {
      if (nextProps.restartWizard === true) {
        this.setState({
          visibleSlide: 0
        });
      }
    }
  }, {
    key: "next",
    value: function next() {
      this.setState({
        visibleSlide: this.state.visibleSlide + 1
      });
      var msg = showInfoMessage(null);
      this.props.bus.send(msg.type, msg);
      this.props.bus.send(RESTART_WIZARD, {
        restartWizard: false
      });
    }
  }, {
    key: "prev",
    value: function prev() {
      this.setState({
        visibleSlide: this.state.visibleSlide - 1
      });
    }
  }, {
    key: "getSlide",
    value: function getSlide(slideNumber) {
      return this.slides[slideNumber];
    }
  }, {
    key: "createProps",
    value: function createProps() {
      var initialProps = {
        connectionStorage: this.state.connectionStorage,
        connectionName: this.props.connectionName
      };
      console.info('createProps', initialProps);
      return initialProps;
    }
  }, {
    key: "render",
    value: function render() {
      console.info("rendering ... ", this.state.visibleSlide, this.props.restartWizard);
      var Slide = this.getSlide(this.state.visibleSlide);
      var props = this.createProps();
      return /*#__PURE__*/react_default.a.createElement(StyledFrame, null, /*#__PURE__*/react_default.a.createElement(SlideContainer, {
        style: {
          padding: '0px'
        }
      }, /*#__PURE__*/react_default.a.createElement(Slide, CarouselMapping_extends({}, props, {
        next: this.next.bind(this),
        prev: this.prev.bind(this),
        index: this.state.visibleSlide
      }))));
    }
  }]);
  return CarouselMapping;
}(react["Component"]);
var CarouselMapping_mapStateToProps = function mapStateToProps(state) {
  var mapping = state.mapping;
  console.info("CarouselMapping", mapping);
  return {
    connectionName: mapping.connectionName,
    nodes: mapping.nodes,
    relationships: mapping.relationships,
    mappings: mapping.mappings,
    restartWizard: mapping.restartWizard
  };
};
/* harmony default export */ var mapping_CarouselMapping = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(CarouselMapping_mapStateToProps, null)(CarouselMapping_CarouselMapping)));
// CONCATENATED MODULE: ./modules/Stream/Auth/styled.jsx
function Auth_styled_extends() { Auth_styled_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Auth_styled_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



var styled_StyledConnectionForm = function StyledConnectionForm(props) {
  return /*#__PURE__*/react_default.a.createElement(Grid_default.a, Auth_styled_extends({
    item: true,
    container: true,
    spacing: 24
  }, props, {
    component: "form"
  }));
};
var styled_StyledConnectionAside = function StyledConnectionAside(props) {
  return /*#__PURE__*/react_default.a.createElement(Grid_default.a, Auth_styled_extends({
    item: true,
    container: true,
    direction: "column",
    justify: "space-between",
    spacing: 24,
    xs: 12
  }, props));
};
var styled_StyledConnectionFrame = function StyledConnectionFrame(props) {
  return /*#__PURE__*/react_default.a.createElement(Grid_default.a, Auth_styled_extends({
    container: true,
    spacing: 24
  }, props));
};
var styled_StyledConnectionBodyContainer = function StyledConnectionBodyContainer(props) {
  return /*#__PURE__*/react_default.a.createElement(Grid_default.a, Auth_styled_extends({
    item: true,
    xs: 12
  }, props));
};
// CONCATENATED MODULE: ./modules/Stream/Auth/ConnectForm.jsx
function ConnectForm_typeof(obj) { "@babel/helpers - typeof"; return ConnectForm_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ConnectForm_typeof(obj); }
function ConnectForm_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function ConnectForm_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ConnectForm_ownKeys(Object(source), !0).forEach(function (key) { ConnectForm_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ConnectForm_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function ConnectForm_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function ConnectForm_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ConnectForm_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ConnectForm_createClass(Constructor, protoProps, staticProps) { if (protoProps) ConnectForm_defineProperties(Constructor.prototype, protoProps); if (staticProps) ConnectForm_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ConnectForm_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ConnectForm_setPrototypeOf(subClass, superClass); }
function ConnectForm_setPrototypeOf(o, p) { ConnectForm_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ConnectForm_setPrototypeOf(o, p); }
function ConnectForm_createSuper(Derived) { var hasNativeReflectConstruct = ConnectForm_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ConnectForm_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ConnectForm_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ConnectForm_possibleConstructorReturn(this, result); }; }
function ConnectForm_possibleConstructorReturn(self, call) { if (call && (ConnectForm_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ConnectForm_assertThisInitialized(self); }
function ConnectForm_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ConnectForm_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ConnectForm_getPrototypeOf(o) { ConnectForm_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ConnectForm_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */












var defaults = {
  'postgresql': {
    port: '5432',
    embeddedDriver: true,
    schemaType: 'value'
  },
  'mysql': {
    port: '3306',
    embeddedDriver: true,
    schemaType: 'custom',
    schemaFieldIsVisible: false,
    getSchemaValue: function getSchemaValue(connectionInfo) {
      return connectionInfo["database"];
    }
  },
  'mssql': {
    port: '1433',
    embeddedDriver: false,
    schemaType: 'value'
  },
  'oracle': {
    port: '1521',
    embeddedDriver: false,
    schemaType: 'custom',
    schemaFieldIsVisible: true,
    getSchemaValue: function getSchemaValue(connectionInfo) {
      var regex = /[;,\s\t]/g;
      return connectionInfo["username"].replace(regex, '').toUpperCase();
    }
  },
  'db2': {
    port: '5021',
    embeddedDriver: false,
    schemaType: 'value'
  },
  'jdbc': {
    port: 'port',
    embeddedDriver: false,
    schemaType: 'value'
  }
};
var ConnectForm_ConnectForm = /*#__PURE__*/function (_Component) {
  ConnectForm_inherits(ConnectForm, _Component);
  var _super = ConnectForm_createSuper(ConnectForm);
  function ConnectForm(props) {
    var _this;
    ConnectForm_classCallCheck(this, ConnectForm);
    _this = _super.call(this, props);
    var connection = props.connection;
    var connectionInfo = {
      name: '',
      host: 'localhost',
      port: defaults['postgresql'].port,
      database: '',
      schema: '',
      defaultSchema: true,
      //only for oracle
      schemaType: defaults['postgresql'].schemaType,
      username: '',
      password: '',
      driverType: 'postgresql',
      embeddedDriver: defaults['postgresql'].embeddedDriver,
      connectionURL: ''
    };
    if (connection) {
      connectionInfo = ConnectForm_objectSpread(ConnectForm_objectSpread({}, connectionInfo), connection);
    }
    _this.state = {
      connecting: false,
      urlConnectionCustom: false,
      connectionInfo: connectionInfo
    };
    return _this;
  }
  ConnectForm_createClass(ConnectForm, [{
    key: "onSaveClick",
    value: function onSaveClick() {
      var _this2 = this;
      var myBus = this.props.bus;
      var connectionInfo = this.state.connectionInfo;
      var etl = new services_EtlToolService(this.state.connectionInfo);
      var processListener = new services_ProcessListener();
      processListener.setSuccessHandler(function (_ref) {
        var stdout = _ref.stdout,
          stderror = _ref.stderror;
        var path = connectionInfo.path || [''];
        var filePathSplit = path.length ? path[0].split(/\\|\//).filter(function (e) {
          return e !== '';
        }) : [''];
        var fileName = filePathSplit.pop() || '';
        var datasource = {
          driverType: connectionInfo.driverType,
          embeddedDriver: connectionInfo.embeddedDriver,
          fileName: fileName
        };
        console.log('Sending %o to metrics', datasource);
        window.neo4jDesktopApi.sendMetrics('neo4j-etl', 'datasource', datasource);
        var outMsg = 'Connection ' + connectionInfo.name + ' saved. ' + stdout;
        var msg = showInfoMessage(outMsg);
        _this2.props.projectStore.currentProject.addSourceConnection(connectionInfo);
        myBus.send(msg.type, msg);
        _this2.props.appendOutput(stderror + outMsg);

        // Remove current frame
        myBus.send(REMOVE, {
          id: 'RDBMS'
        });
        myBus.send(CONNECTION_UPDATE, {
          connectionInfo: connectionInfo
        });

        // Add connection frame
        myBus.send(ADD, {
          state: {
            id: 'MAPPING',
            type: 'mapping'
          }
        });
      });
      processListener.setFailureHandler(function (error) {
        console.error(error);
        var actErr = showErrorMessage(error);
        myBus.send(actErr.type, actErr);
        _this2.props.appendOutput(error);
      });
      processListener.setErrorListener(function (error) {
        if (error.message.indexOf('Java is restricted to execute') !== -1) {
          myBus.send(JAVA_PERMISSION_ERROR_SHOW);
        }
      });
      etl.testConnection(processListener);
    }
  }, {
    key: "onInputChange",
    value: function onInputChange(e) {
      var target = e.target;
      var name = target.name;
      var value = target.value;
      var myBus = this.props.bus;
      if (name === 'connectionURL') {
        this.setState(function (state) {
          return {
            urlConnectionCustom: !!value,
            connectionInfo: ConnectForm_objectSpread(ConnectForm_objectSpread({}, state.connectionInfo), {}, {
              connectionURL: value
            })
          };
        });
        return;
      }
      this.setState(function (_ref2) {
        var connectionInfo = _ref2.connectionInfo,
          urlConnectionCustom = _ref2.urlConnectionCustom;
        connectionInfo[name] = value;
        if (name === 'driverType') {
          connectionInfo['port'] = defaults[value].port;
          connectionInfo['embeddedDriver'] = defaults[value].embeddedDriver;
        }
        if (name === 'driverPath') {
          var files = [];
          for (var i = 0, len = target.files.length; i < len; i++) {
            files.push(target.files[i].path);
          }
          connectionInfo['path'] = files; //e.target.files[0].path
        }

        if (name === 'defaultSchema') {
          connectionInfo['defaultSchema'] = target.checked;
        }
        var regex = /[;,\s\t]/g;
        if (name === 'schema') {
          if (!value.match(regex)) {
            connectionInfo['schema'] = value;
          } else {
            connectionInfo['schema'] = value.replace(regex, '');
            var error = 'Invalid schema: only one schema is allowed. Also chars ", ;" and spaces/tabs are not permitted';
            console.info(error);
            var actErr = showErrorMessage(error);
            myBus.send(actErr.type, actErr);
          }
        }
        var url = urlConnection(connectionInfo.driverType, connectionInfo.host, connectionInfo.port, connectionInfo.database);
        connectionInfo['connectionURL'] = urlConnectionCustom ? connectionInfo.connectionURL : url;
        connectionInfo['schemaType'] = defaults[connectionInfo["driverType"]].schemaType;
        connectionInfo['getSchemaValue'] = defaults[connectionInfo["driverType"]].getSchemaValue;
        connectionInfo['schemaFieldIsVisible'] = defaults[connectionInfo["driverType"]].schemaFieldIsVisible || false;
        if (connectionInfo['schemaType'] === 'custom') {
          if (connectionInfo['getSchemaValue'] && connectionInfo['defaultSchema']) {
            connectionInfo['schema'] = connectionInfo['getSchemaValue'](connectionInfo);
          }
        }
        if (connectionInfo['schemaType'] === 'default') {
          if (connectionInfo['defaultSchema']) {
            connectionInfo['schema'] = connectionInfo["username"].replace(regex, '').toUpperCase();
          }
        }
        if (connectionInfo['schemaType'] === 'none') {
          connectionInfo['schema'] = '';
        }
        return {
          connectionInfo: connectionInfo
        };
      });
    }
  }, {
    key: "render",
    value: function render() {
      return /*#__PURE__*/react_default.a.createElement(styled_StyledConnectionForm, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Connection Name",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "name",
        value: this.state.connectionInfo.name
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 9
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Host",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "host",
        value: this.state.connectionInfo.host
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 3
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Port",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "port",
        value: this.state.connectionInfo.port
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Type",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "driverType",
        value: this.state.connectionInfo.driverType,
        select: true
      }, Object.keys(defaults).map(function (item, index) {
        return /*#__PURE__*/react_default.a.createElement(MenuItem_default.a, {
          key: index,
          value: item
        }, item);
      }))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Database",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "database",
        value: this.state.connectionInfo.database
      })), /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.state.connectionInfo.schemaType === 'custom' && this.state.connectionInfo.schemaFieldIsVisible === true
      }, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Schema",
        onChange: this.onInputChange.bind(this),
        name: "schema",
        fullWidth: true,
        value: this.state.connectionInfo.schema,
        disabled: this.state.connectionInfo.defaultSchema
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(FormControlLabel_default.a, {
        control: /*#__PURE__*/react_default.a.createElement(Checkbox_default.a, {
          name: "defaultSchema",
          checked: this.state.connectionInfo.defaultSchema,
          onChange: this.onInputChange.bind(this)
        }),
        label: "Default Value"
      }))), /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.state.connectionInfo.schemaType === 'value'
      }, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Schema",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "schema",
        value: this.state.connectionInfo.schema
      }))), /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.state.connectionInfo.embeddedDriver === false
      }, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "JDBC driver path",
        fullWidth: true,
        name: "driverPath",
        inputProps: {
          multiple: "true",
          type: "file",
          accept: ".jar"
        },
        onChange: this.onInputChange.bind(this),
        InputLabelProps: {
          shrink: true
        }
      }))), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 12
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Connection URL",
        fullWidth: true,
        name: "connectionURL",
        value: this.state.connectionInfo.connectionURL,
        onChange: this.onInputChange.bind(this)
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Username",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "username",
        value: this.state.connectionInfo.username
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, /*#__PURE__*/react_default.a.createElement(TextField_default.a, {
        label: "Password",
        fullWidth: true,
        onChange: this.onInputChange.bind(this),
        name: "password",
        value: this.state.connectionInfo.password,
        type: "password"
      })), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6
      }, this.props.backButtom), /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
        item: true,
        xs: 6,
        style: {
          textAlign: 'right'
        }
      }, /*#__PURE__*/react_default.a.createElement(Button_default.a, {
        onClick: this.onSaveClick.bind(this),
        variant: "outlined"
      }, "Test and Save Connection")));
    }
  }]);
  return ConnectForm;
}(react["Component"]);
var ConnectForm_mapStateToProps = function mapStateToProps(state) {
  return {
    projectStore: state.projectStore
  };
};
/* harmony default export */ var Auth_ConnectForm = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(ConnectForm_mapStateToProps, null)(ConnectForm_ConnectForm)));
// CONCATENATED MODULE: ./modules/Stream/Auth/ConnectionForm.jsx
function ConnectionForm_typeof(obj) { "@babel/helpers - typeof"; return ConnectionForm_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ConnectionForm_typeof(obj); }
function ConnectionForm_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function ConnectionForm_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ConnectionForm_ownKeys(Object(source), !0).forEach(function (key) { ConnectionForm_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ConnectionForm_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function ConnectionForm_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function ConnectionForm_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ConnectionForm_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ConnectionForm_createClass(Constructor, protoProps, staticProps) { if (protoProps) ConnectionForm_defineProperties(Constructor.prototype, protoProps); if (staticProps) ConnectionForm_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ConnectionForm_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ConnectionForm_setPrototypeOf(subClass, superClass); }
function ConnectionForm_setPrototypeOf(o, p) { ConnectionForm_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ConnectionForm_setPrototypeOf(o, p); }
function ConnectionForm_createSuper(Derived) { var hasNativeReflectConstruct = ConnectionForm_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ConnectionForm_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ConnectionForm_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ConnectionForm_possibleConstructorReturn(this, result); }; }
function ConnectionForm_possibleConstructorReturn(self, call) { if (call && (ConnectionForm_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ConnectionForm_assertThisInitialized(self); }
function ConnectionForm_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ConnectionForm_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ConnectionForm_getPrototypeOf(o) { ConnectionForm_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ConnectionForm_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */







var ConnectionForm_ConnectionForm = /*#__PURE__*/function (_Component) {
  ConnectionForm_inherits(ConnectionForm, _Component);
  var _super = ConnectionForm_createSuper(ConnectionForm);
  function ConnectionForm(props) {
    var _this;
    ConnectionForm_classCallCheck(this, ConnectionForm);
    _this = _super.call(this, props);
    var connection = _this.props.connection;
    var isConnected = !!props.activeConnection;
    _this.state = ConnectionForm_objectSpread(ConnectionForm_objectSpread({}, connection), {}, {
      isConnected: isConnected,
      successCallback: props.onSuccess || function () {},
      used: isConnected,
      connection: connection
    });
    return _this;
  }
  ConnectionForm_createClass(ConnectionForm, [{
    key: "connect",
    value: function connect() {
      this.props.error({});
    }
  }, {
    key: "onUsernameChange",
    value: function onUsernameChange(event) {
      var username = event.target.value;
      this.setState({
        username: username
      });
      this.props.error({});
    }
  }, {
    key: "onPasswordChange",
    value: function onPasswordChange(event) {
      var password = event.target.value;
      this.setState({
        password: password
      });
      this.props.error({});
    }
  }, {
    key: "onHostChange",
    value: function onHostChange(event) {
      var host = event.target.value;
      this.setState({
        host: host,
        hostInputVal: host
      });
      this.props.error({});
    }
  }, {
    key: "saveAndStart",
    value: function saveAndStart() {
      this.setState({
        used: true
      });
      this.state.successCallback();
      this.saveCredentials();
      this.props.setActiveConnection(this.state.id);
      this.props.executeInitCmd();
    }
  }, {
    key: "saveCredentials",
    value: function saveCredentials() {
      this.props.updateConnection({
        id: this.state.id,
        host: this.state.host,
        username: this.state.username,
        password: this.state.password
      });
    }
  }, {
    key: "componentWillReceiveProps",
    value: function componentWillReceiveProps(nextProps) {
      if (nextProps.oldPassword) {
        this.setState({
          oldPassword: nextProps.oldPassword
        });
      }
      if (nextProps.activeConnection && nextProps.activeConnectionData) {
        this.setState({
          isConnected: true
        });
      } else {
        this.setState({
          isConnected: false
        });
      }
    }
  }, {
    key: "render",
    value: function render() {
      var edit = this.props.edit ? true : false;
      return /*#__PURE__*/react_default.a.createElement(Auth_ConnectForm, {
        onConnectClick: this.connect.bind(this),
        onHostChange: this.onHostChange.bind(this),
        onUsernameChange: this.onUsernameChange.bind(this),
        onPasswordChange: this.onPasswordChange.bind(this),
        host: this.state.hostInputVal || this.state.host,
        username: this.state.username,
        password: this.state.password,
        used: this.state.used,
        appendOutput: this.props.appendOutput,
        backButtom: this.props.backButtom,
        connection: this.state.connection,
        edit: edit
      });
    }
  }]);
  return ConnectionForm;
}(react["Component"]);
var ConnectionForm_mapStateToProps = function mapStateToProps(state) {
  return {
    initCmd: getInitCmd(state),
    activeConnection: {},
    activeConnectionData: {},
    storeCredentials: {}
  };
};
var ConnectionForm_mapDispatchToProps = function mapDispatchToProps(dispatch) {
  return {
    updateConnection: function updateConnection() {},
    setActiveConnection: function setActiveConnection() {},
    dispatchInitCmd: function dispatchInitCmd(initCmd) {
      return dispatch(executeSystemCommand(initCmd));
    }
  };
};
var mergeProps = function mergeProps(stateProps, dispatchProps, ownProps) {
  return ConnectionForm_objectSpread(ConnectionForm_objectSpread(ConnectionForm_objectSpread({
    activeConnection: stateProps.activeConnection,
    activeConnectionData: stateProps.activeConnectionData,
    storeCredentials: stateProps.storeCredentials
  }, ownProps), dispatchProps), {}, {
    executeInitCmd: function executeInitCmd() {
      dispatchProps.dispatchInitCmd(stateProps.initCmd);
    }
  });
};
/* harmony default export */ var Auth_ConnectionForm = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(ConnectionForm_mapStateToProps, ConnectionForm_mapDispatchToProps, mergeProps)(ConnectionForm_ConnectionForm)));
// EXTERNAL MODULE: ./components/icons/style.css
var icons_style = __webpack_require__("./components/icons/style.css");
var icons_style_default = /*#__PURE__*/__webpack_require__.n(icons_style);

// CONCATENATED MODULE: ./components/icons/Icons.jsx
function Icons_typeof(obj) { "@babel/helpers - typeof"; return Icons_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, Icons_typeof(obj); }
var Icons_excluded = ["activeStyle", "inactiveStyle", "isOpen", "text", "regulateSize"];
function Icons_extends() { Icons_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Icons_extends.apply(this, arguments); }
function Icons_objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = Icons_objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }
function Icons_objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }
function Icons_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function Icons_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function Icons_createClass(Constructor, protoProps, staticProps) { if (protoProps) Icons_defineProperties(Constructor.prototype, protoProps); if (staticProps) Icons_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function Icons_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) Icons_setPrototypeOf(subClass, superClass); }
function Icons_setPrototypeOf(o, p) { Icons_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return Icons_setPrototypeOf(o, p); }
function Icons_createSuper(Derived) { var hasNativeReflectConstruct = Icons_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = Icons_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = Icons_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return Icons_possibleConstructorReturn(this, result); }; }
function Icons_possibleConstructorReturn(self, call) { if (call && (Icons_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return Icons_assertThisInitialized(self); }
function Icons_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function Icons_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function Icons_getPrototypeOf(o) { Icons_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return Icons_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */




var Icons_IconContainer = /*#__PURE__*/function (_Component) {
  Icons_inherits(IconContainer, _Component);
  var _super = Icons_createSuper(IconContainer);
  function IconContainer(props) {
    var _this;
    Icons_classCallCheck(this, IconContainer);
    _this = _super.call(this, props);
    _this.state = {
      mouseover: false
    };
    return _this;
  }
  Icons_createClass(IconContainer, [{
    key: "mouseover",
    value: function mouseover() {
      this.setState({
        mouseover: true
      });
    }
  }, {
    key: "mouseout",
    value: function mouseout() {
      this.setState({
        mouseover: false
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this$props = this.props,
        activeStyle = _this$props.activeStyle,
        inactiveStyle = _this$props.inactiveStyle,
        isOpen = _this$props.isOpen,
        text = _this$props.text,
        regulateSize = _this$props.regulateSize,
        rest = Icons_objectWithoutProperties(_this$props, Icons_excluded);
      var state = this.state.mouseover || isOpen ? activeStyle || '' : inactiveStyle || '';
      var newClass = this.props.suppressIconStyles ? this.props.className : state + ' ' + this.props.className;
      var regulateSizeStyle = regulateSize ? {
        'font-size': regulateSize + 'em'
      } : null;
      var icon = /*#__PURE__*/react_default.a.createElement("i", Icons_extends({}, rest, {
        className: newClass,
        onMouseEnter: this.mouseover.bind(this),
        onMouseLeave: this.mouseout.bind(this),
        style: regulateSizeStyle
      }));
      return text ? /*#__PURE__*/react_default.a.createElement("span", null, icon, /*#__PURE__*/react_default.a.createElement(StyledText, null, text)) : icon;
    }
  }]);
  return IconContainer;
}(react["Component"]);
var StyledText = styled_components_es["default"].div.withConfig({
  displayName: "Icons__StyledText",
  componentId: "sc-hi5o15-0"
})(["font-size:9px;line-height:10px;margin-top:4px;padding:0;"]);
var Icons_DatabaseIcon = function DatabaseIcon(_ref) {
  var isOpen = _ref.isOpen;
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    isOpen: isOpen,
    activeStyle: icons_style_default.a.white,
    inactiveStyle: icons_style_default.a.inactive,
    className: "sl sl-database"
  });
};
var Icons_AboutIcon = function AboutIcon(_ref2) {
  var isOpen = _ref2.isOpen;
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    isOpen: isOpen,
    activeStyle: icons_style_default.a.credits,
    inactiveStyle: icons_style_default.a.inactive,
    className: "nw nw-neo4j-outline-32px"
  });
};
var Icons_AlertIcon = function AlertIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "sl-alert",
    text: "Warn"
  });
};
var Icons_ErrorIcon = function ErrorIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "fa fa-file-text-o",
    text: "Error"
  });
};
var Icons_BinIcon = function BinIcon(props) {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, Icons_extends({
    activeStyle: icons_style_default.a.white,
    inactiveStyle: icons_style_default.a.inactive
  }, props, {
    className: "sl-bin"
  }));
};
var Icons_ExpandIcon = function ExpandIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "sl-scale-spread"
  });
};
var Icons_ContractIcon = function ContractIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "sl-scale-reduce"
  });
};
var Icons_RefreshIcon = function RefreshIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "sl-loop"
  });
};
var Icons_CloseIcon = function CloseIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "sl-delete",
    regulateSize: "0.85"
  });
};
var Icons_MinusIcon = function MinusIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.blue,
    inactiveStyle: icons_style_default.a.inactive,
    className: "sl-minus-circle"
  });
};
var Icons_RightArrowIcon = function RightArrowIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.white,
    inactiveStyle: icons_style_default.a.inactive,
    className: "sl-arrow-circle-right"
  });
};
var Icons_CancelIcon = function CancelIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.blue,
    inactiveStyle: icons_style_default.a.inactive,
    className: "sl-delete-circle"
  });
};
var Icons_PlayIcon = function PlayIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.lightBlue,
    inactiveStyle: icons_style_default.a.blue,
    className: "fa fa-play-circle-o"
  });
};
var Icons_PlainPlayIcon = function PlainPlayIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    className: "fa fa-play-circle"
  });
};
var Icons_QuestionIcon = function QuestionIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.lightBlue,
    inactiveStyle: icons_style_default.a.blue,
    className: "fa fa-question-circle-o"
  });
};
var Icons_PlusIcon = function PlusIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.white,
    inactiveStyle: icons_style_default.a.white,
    className: "fa fa-plus"
  });
};
var Icons_EditIcon = function EditIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    activeStyle: icons_style_default.a.white,
    inactiveStyle: icons_style_default.a.white,
    className: "sl-pencil"
  });
};
var Icons_ExclamationTriangleIcon = function ExclamationTriangleIcon() {
  return /*#__PURE__*/react_default.a.createElement(Icons_IconContainer, {
    suppressIconStyles: true,
    className: "fa fa-exclamation-triangle"
  });
};
// CONCATENATED MODULE: ./components/Ellipsis.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


var Ellipsis = styled_components_es["default"].div.withConfig({
  displayName: "Ellipsis",
  componentId: "sc-yi29dc-0"
})(["text-overflow:ellipsis;overflow:hidden;white-space:nowrap;"]);
/* harmony default export */ var components_Ellipsis = (Ellipsis);
// CONCATENATED MODULE: ./modules/Stream/FrameError.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */





var FrameError_FrameError = function FrameError(props) {
  if (!props || !props.code && !props.message) return null;
  var fullError = "".concat(props.code).concat(props.message ? ':' : '', " ").concat(props.message || '');
  return /*#__PURE__*/react_default.a.createElement(components_Ellipsis, null, /*#__PURE__*/react_default.a.createElement(ErrorText, {
    title: fullError
  }, /*#__PURE__*/react_default.a.createElement(Icons_ExclamationTriangleIcon, null), " ", fullError));
};
/* harmony default export */ var Stream_FrameError = (FrameError_FrameError);
// CONCATENATED MODULE: ./modules/Stream/Auth/ConnectionFrame.jsx
function ConnectionFrame_typeof(obj) { "@babel/helpers - typeof"; return ConnectionFrame_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, ConnectionFrame_typeof(obj); }
function ConnectionFrame_extends() { ConnectionFrame_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return ConnectionFrame_extends.apply(this, arguments); }
function ConnectionFrame_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function ConnectionFrame_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function ConnectionFrame_createClass(Constructor, protoProps, staticProps) { if (protoProps) ConnectionFrame_defineProperties(Constructor.prototype, protoProps); if (staticProps) ConnectionFrame_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function ConnectionFrame_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) ConnectionFrame_setPrototypeOf(subClass, superClass); }
function ConnectionFrame_setPrototypeOf(o, p) { ConnectionFrame_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return ConnectionFrame_setPrototypeOf(o, p); }
function ConnectionFrame_createSuper(Derived) { var hasNativeReflectConstruct = ConnectionFrame_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = ConnectionFrame_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = ConnectionFrame_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return ConnectionFrame_possibleConstructorReturn(this, result); }; }
function ConnectionFrame_possibleConstructorReturn(self, call) { if (call && (ConnectionFrame_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return ConnectionFrame_assertThisInitialized(self); }
function ConnectionFrame_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function ConnectionFrame_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function ConnectionFrame_getPrototypeOf(o) { ConnectionFrame_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return ConnectionFrame_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */













var ConnectionFrame_ConnectionFrame = /*#__PURE__*/function (_Component) {
  ConnectionFrame_inherits(ConnectionFrame, _Component);
  var _super = ConnectionFrame_createSuper(ConnectionFrame);
  function ConnectionFrame(props) {
    var _this;
    ConnectionFrame_classCallCheck(this, ConnectionFrame);
    _this = _super.call(this, props);
    _this.state = {
      error: {},
      output: '',
      showLogs: false
    };
    _this.appendOutput = _this.appendOutput.bind(ConnectionFrame_assertThisInitialized(_this));
    _this.toggleLogs = _this.toggleLogs.bind(ConnectionFrame_assertThisInitialized(_this));
    return _this;
  }
  ConnectionFrame_createClass(ConnectionFrame, [{
    key: "error",
    value: function error(e) {
      this.setState({
        error: e
      });
    }
  }, {
    key: "success",
    value: function success() {
      this.setState({
        success: true
      });
    }
  }, {
    key: "backToStart",
    value: function backToStart(event) {
      event.preventDefault();
      this.props.bus.send(RESTART_WIZARD, {
        restartWizard: true
      });

      // Remove current frame
      this.props.bus.send(REMOVE, {
        id: 'RDBMS'
      });

      // Add connection frame
      this.props.bus.send(ADD, {
        state: {
          id: 'MAPPING',
          type: 'mapping'
        }
      });
    }
  }, {
    key: "appendOutput",
    value: function appendOutput(output) {
      this.setState(function (state) {
        return {
          output: state.output + output
        };
      });
    }
  }, {
    key: "toggleLogs",
    value: function toggleLogs() {
      this.setState(function (state) {
        return {
          showLogs: !state.showLogs
        };
      });
    }
  }, {
    key: "render",
    value: function render() {
      var connection = this.props.frame.connection;
      var edit = false;
      if (connection) {
        edit = true;
      }
      var backButtom = /*#__PURE__*/react_default.a.createElement(buttons_FormButton, {
        onClick: this.backToStart.bind(this),
        label: "Back to Start"
      });
      return /*#__PURE__*/react_default.a.createElement(Stream_FrameTemplate, {
        header: this.props.frame,
        statusbar: /*#__PURE__*/react_default.a.createElement(Stream_FrameError, {
          code: this.state.error.code,
          message: this.state.error.message
        }),
        contents: /*#__PURE__*/react_default.a.createElement(styled_StyledConnectionFrame, null, /*#__PURE__*/react_default.a.createElement(styled_StyledConnectionAside, null, /*#__PURE__*/react_default.a.createElement(Grid_default.a, {
          item: true,
          xs: 12
        }, /*#__PURE__*/react_default.a.createElement(Headers_H3, null, edit ? "Edit JDBC Connection: ".concat(connection.name) : 'JDBC Connection'), /*#__PURE__*/react_default.a.createElement(Text_Lead, null, "Database access requires an authenticated connection and a valid JDBC driver."))), /*#__PURE__*/react_default.a.createElement(styled_StyledConnectionBodyContainer, null, /*#__PURE__*/react_default.a.createElement(Auth_ConnectionForm, ConnectionFrame_extends({}, this.props, {
          edit: edit,
          connection: connection,
          onSuccess: this.success.bind(this),
          error: this.error.bind(this),
          appendOutput: this.appendOutput,
          backButtom: backButtom
        }))))
      });
    }
  }]);
  return ConnectionFrame;
}(react["Component"]);
/* harmony default export */ var Auth_ConnectionFrame = (Object(react_suber_lib["withBus"])(ConnectionFrame_ConnectionFrame));
// CONCATENATED MODULE: ./modules/Stream/Stream.jsx
function Stream_typeof(obj) { "@babel/helpers - typeof"; return Stream_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, Stream_typeof(obj); }
function Stream_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function Stream_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function Stream_createClass(Constructor, protoProps, staticProps) { if (protoProps) Stream_defineProperties(Constructor.prototype, protoProps); if (staticProps) Stream_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function Stream_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) Stream_setPrototypeOf(subClass, superClass); }
function Stream_setPrototypeOf(o, p) { Stream_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return Stream_setPrototypeOf(o, p); }
function Stream_createSuper(Derived) { var hasNativeReflectConstruct = Stream_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = Stream_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = Stream_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return Stream_possibleConstructorReturn(this, result); }; }
function Stream_possibleConstructorReturn(self, call) { if (call && (Stream_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return Stream_assertThisInitialized(self); }
function Stream_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function Stream_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function Stream_getPrototypeOf(o) { Stream_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return Stream_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */













var Stream_getFrame = function getFrame(type) {
  var trans = {
    error: Stream_ErrorFrame,
    pre: Stream_PreFrame,
    connection: Auth_ConnectionFrame,
    help: Stream_HelpFrame,
    mapping: mapping_CarouselMapping,
    "default": Stream_Frame
  };
  return trans[type] || trans["default"];
};
var Stream_Stream = /*#__PURE__*/function (_Component) {
  Stream_inherits(Stream, _Component);
  var _super = Stream_createSuper(Stream);
  function Stream() {
    Stream_classCallCheck(this, Stream);
    return _super.apply(this, arguments);
  }
  Stream_createClass(Stream, [{
    key: "shouldComponentUpdate",
    value: function shouldComponentUpdate(props) {
      var frameHasBeenAdded = this.props.frames.length < props.frames.length;
      if (frameHasBeenAdded && this.props.scrollToTop) {
        if (this.base) {
          this.base.scrollTop = 0;
        }
        return true;
      }
      return !deepEquals(props, this.props);
    }
  }, {
    key: "render",
    value: function render() {
      var _this = this;
      return /*#__PURE__*/react_default.a.createElement(StyledStream, null, this.props.frames.map(function (frame) {
        var frameProps = {
          frame: frame,
          activeConnectionData: _this.props.activeConnectionData
        };
        var MyFrame = Stream_getFrame(frame.type);
        return /*#__PURE__*/react_default.a.createElement("div", {
          key: frame.id
        }, /*#__PURE__*/react_default.a.createElement(MyFrame, frameProps));
      }));
    }
  }]);
  return Stream;
}(react["Component"]);
var Stream_mapStateToProps = function mapStateToProps(state) {
  return {
    frames: getFrames(state),
    activeConnectionData: {},
    scrollToTop: getScrollToTop(state)
  };
};
/* harmony default export */ var modules_Stream_Stream = (Object(react_redux_es["connect"])(Stream_mapStateToProps)(Stream_Stream));
// CONCATENATED MODULE: ./modules/Main/styled.jsx
function Main_styled_extends() { Main_styled_extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return Main_styled_extends.apply(this, arguments); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */





var styled_grow = function grow(height) {
  return Object(styled_components_es["keyframes"])(["0%{max-height:0px;}100%{max-height:", ";}"], height);
};
var StyledMain = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledMain",
  componentId: "sc-1fcte9t-0"
})(["flex:0 0 auto;display:flex;flex-direction:column;min-width:0;position:relative;height:100vh;"]);
var Banner = styled_components_es["default"].div.withConfig({
  displayName: "styled__Banner",
  componentId: "sc-1fcte9t-1"
})(["color:white;padding:18px 24px;margin:0;animation:", " 0.3s ease-in;position:absolute;top:0;left:0;width:100%;z-index:100;"], styled_grow('49px'));
var BannerPre = styled_components_es["default"].pre.withConfig({
  displayName: "styled__BannerPre",
  componentId: "sc-1fcte9t-2"
})(["overflow:auto;max-height:100px;padding:0;margin:0;font-family:inherit;font-size:inherit;max-width:calc(100% - 8px);"]);
var ErrorBanner = Object(styled_components_es["default"])(Banner).withConfig({
  displayName: "styled__ErrorBanner",
  componentId: "sc-1fcte9t-3"
})(["background-color:", ";"], function (props) {
  return props.theme.error;
});
var InfoBanner = Object(styled_components_es["default"])(Banner).withConfig({
  displayName: "styled__InfoBanner",
  componentId: "sc-1fcte9t-4"
})(["background-color:", ";"], function (props) {
  return props.theme.auth;
});
var styled_CloseBanner = function CloseBanner(props) {
  return /*#__PURE__*/react_default.a.createElement(Icon_default.a, Main_styled_extends({}, props, {
    style: {
      position: 'absolute',
      top: '10px',
      right: '10px',
      cursor: 'pointer'
    }
  }), "close");
};
var SyncDisconnectedBanner = Object(styled_components_es["default"])(Banner).withConfig({
  displayName: "styled__SyncDisconnectedBanner",
  componentId: "sc-1fcte9t-5"
})(["background-color:", ";display:flex;justify-content:space-between;"], function (props) {
  return props.theme.auth;
});
var SyncSignInBarButton = Object(styled_components_es["default"])(SyncSignInButton).withConfig({
  displayName: "styled__SyncSignInBarButton",
  componentId: "sc-1fcte9t-6"
})(["padding:0 8px 0 8px;margin:0 12px 0 12px;vertical-align:baseline;border:none;"]);
var StyledCancelLink = styled_components_es["default"].a.withConfig({
  displayName: "styled__StyledCancelLink",
  componentId: "sc-1fcte9t-7"
})(["cursor:pointer;text-decoration:none;color:#d0d0d0;&:hover{color:#ffffff;text-decoration:none;}"]);
var StyledSyncReminderSpan = styled_components_es["default"].span.withConfig({
  displayName: "styled__StyledSyncReminderSpan",
  componentId: "sc-1fcte9t-8"
})(["text-overflow:ellipsis;white-space:nowrap;overflow:hidden;float:left;"]);
var StyledSyncReminderButtonContainer = styled_components_es["default"].div.withConfig({
  displayName: "styled__StyledSyncReminderButtonContainer",
  componentId: "sc-1fcte9t-9"
})(["float:right;"]);
// CONCATENATED MODULE: ./modules/Main/Main.jsx
function Main_typeof(obj) { "@babel/helpers - typeof"; return Main_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, Main_typeof(obj); }
function Main_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function Main_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function Main_createClass(Constructor, protoProps, staticProps) { if (protoProps) Main_defineProperties(Constructor.prototype, protoProps); if (staticProps) Main_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function Main_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) Main_setPrototypeOf(subClass, superClass); }
function Main_setPrototypeOf(o, p) { Main_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return Main_setPrototypeOf(o, p); }
function Main_createSuper(Derived) { var hasNativeReflectConstruct = Main_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = Main_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = Main_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return Main_possibleConstructorReturn(this, result); }; }
function Main_possibleConstructorReturn(self, call) { if (call && (Main_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return Main_assertThisInitialized(self); }
function Main_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function Main_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function Main_getPrototypeOf(o) { Main_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return Main_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */







var Main_Main = /*#__PURE__*/function (_React$Component) {
  Main_inherits(Main, _React$Component);
  var _super = Main_createSuper(Main);
  function Main(props) {
    var _this;
    Main_classCallCheck(this, Main);
    _this = _super.call(this, props);
    _this.emptyErrorMessage = _this.emptyErrorMessage.bind(Main_assertThisInitialized(_this));
    _this.emptyInfoMessage = _this.emptyInfoMessage.bind(Main_assertThisInitialized(_this));
    return _this;
  }
  Main_createClass(Main, [{
    key: "emptyErrorMessage",
    value: function emptyErrorMessage() {
      this.props.bus.send(SHOW_ERROR_MESSAGE, '');
    }
  }, {
    key: "emptyInfoMessage",
    value: function emptyInfoMessage() {
      this.props.bus.send(SHOW_INFO_MESSAGE, '');
    }
  }, {
    key: "render",
    value: function render() {
      return /*#__PURE__*/react_default.a.createElement(StyledMain, null, /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.props.errorMessage
      }, /*#__PURE__*/react_default.a.createElement(ErrorBanner, null, /*#__PURE__*/react_default.a.createElement(BannerPre, null, this.props.errorMessage), /*#__PURE__*/react_default.a.createElement(styled_CloseBanner, {
        onClick: this.emptyErrorMessage
      }))), /*#__PURE__*/react_default.a.createElement(components_Render, {
        "if": this.props.infoMessage
      }, /*#__PURE__*/react_default.a.createElement(InfoBanner, null, /*#__PURE__*/react_default.a.createElement(BannerPre, null, this.props.infoMessage), /*#__PURE__*/react_default.a.createElement(styled_CloseBanner, {
        onClick: this.emptyInfoMessage
      }))), /*#__PURE__*/react_default.a.createElement(modules_Stream_Stream, null));
    }
  }]);
  return Main;
}(react_default.a.Component);
/* harmony default export */ var modules_Main_Main = (Object(react_suber_lib["withBus"])(Main_Main));
// CONCATENATED MODULE: ./modules/DocTitle/index.jsx
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var DocTitle = function DocTitle(_ref) {
  var titleString = _ref.titleString;
  document.title = titleString;
  return null;
};
/* harmony default export */ var modules_DocTitle = (DocTitle);
// EXTERNAL MODULE: ./modules/Stream/JavaPerm.jpg
var Stream_JavaPerm = __webpack_require__("./modules/Stream/JavaPerm.jpg");
var JavaPerm_default = /*#__PURE__*/__webpack_require__.n(Stream_JavaPerm);

// CONCATENATED MODULE: ./modules/Stream/JavaPerm.jsx
function JavaPerm_typeof(obj) { "@babel/helpers - typeof"; return JavaPerm_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, JavaPerm_typeof(obj); }
function JavaPerm_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function JavaPerm_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function JavaPerm_createClass(Constructor, protoProps, staticProps) { if (protoProps) JavaPerm_defineProperties(Constructor.prototype, protoProps); if (staticProps) JavaPerm_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function JavaPerm_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) JavaPerm_setPrototypeOf(subClass, superClass); }
function JavaPerm_setPrototypeOf(o, p) { JavaPerm_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return JavaPerm_setPrototypeOf(o, p); }
function JavaPerm_createSuper(Derived) { var hasNativeReflectConstruct = JavaPerm_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = JavaPerm_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = JavaPerm_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return JavaPerm_possibleConstructorReturn(this, result); }; }
function JavaPerm_possibleConstructorReturn(self, call) { if (call && (JavaPerm_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return JavaPerm_assertThisInitialized(self); }
function JavaPerm_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function JavaPerm_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function JavaPerm_getPrototypeOf(o) { JavaPerm_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return JavaPerm_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */







var JavaPerm_JavaPerm = /*#__PURE__*/function (_Component) {
  JavaPerm_inherits(JavaPerm, _Component);
  var _super = JavaPerm_createSuper(JavaPerm);
  function JavaPerm(props) {
    var _this;
    JavaPerm_classCallCheck(this, JavaPerm);
    _this = _super.call(this, props);
    _this.handleClose = _this.handleClose.bind(JavaPerm_assertThisInitialized(_this));
    return _this;
  }
  JavaPerm_createClass(JavaPerm, [{
    key: "handleClose",
    value: function handleClose() {
      this.props.bus.send(JAVA_PERMISSION_ERROR_HIDE);
    }
  }, {
    key: "render",
    value: function render() {
      return /*#__PURE__*/react_default.a.createElement(Dialog_default.a, {
        open: this.props.open,
        onClose: this.handleClose,
        "aria-labelledby": "alert-dialog-title",
        "aria-describedby": "alert-dialog-description"
      }, /*#__PURE__*/react_default.a.createElement(DialogTitle_default.a, {
        id: "alert-dialog-title"
      }, "Please permit background process and reload the app:"), /*#__PURE__*/react_default.a.createElement(DialogContent_default.a, {
        style: {
          textAlign: 'center'
        }
      }, /*#__PURE__*/react_default.a.createElement("img", {
        src: JavaPerm_default.a,
        width: "340"
      })), /*#__PURE__*/react_default.a.createElement(DialogActions_default.a, null, /*#__PURE__*/react_default.a.createElement(Button_default.a, {
        onClick: this.handleClose,
        color: "primary",
        autoFocus: true
      }, "Close")));
    }
  }]);
  return JavaPerm;
}(react["Component"]);
var JavaPerm_mapStateToProps = function mapStateToProps(state) {
  return {
    open: state.frames.javaPermErrorShow
  };
};
/* harmony default export */ var modules_Stream_JavaPerm = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(JavaPerm_mapStateToProps, null)(JavaPerm_JavaPerm)));
// CONCATENATED MODULE: ./modules/App/App.jsx
function App_typeof(obj) { "@babel/helpers - typeof"; return App_typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, App_typeof(obj); }
function App_classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function App_defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }
function App_createClass(Constructor, protoProps, staticProps) { if (protoProps) App_defineProperties(Constructor.prototype, protoProps); if (staticProps) App_defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function App_inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) App_setPrototypeOf(subClass, superClass); }
function App_setPrototypeOf(o, p) { App_setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return App_setPrototypeOf(o, p); }
function App_createSuper(Derived) { var hasNativeReflectConstruct = App_isNativeReflectConstruct(); return function _createSuperInternal() { var Super = App_getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = App_getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return App_possibleConstructorReturn(this, result); }; }
function App_possibleConstructorReturn(self, call) { if (call && (App_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return App_assertThisInitialized(self); }
function App_assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function App_isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function App_getPrototypeOf(o) { App_getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return App_getPrototypeOf(o); }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

















var App_App = /*#__PURE__*/function (_Component) {
  App_inherits(App, _Component);
  var _super = App_createSuper(App);
  function App() {
    App_classCallCheck(this, App);
    return _super.apply(this, arguments);
  }
  App_createClass(App, [{
    key: "render",
    value: function render() {
      var _this$props = this.props,
        handleNavClick = _this$props.handleNavClick,
        theme = _this$props.theme,
        javaPermErrorShow = _this$props.javaPermErrorShow;
      var themeData = themes_namespaceObject[theme] || normal;
      return /*#__PURE__*/react_default.a.createElement(styled_components_es["ThemeProvider"], {
        theme: themeData
      }, /*#__PURE__*/react_default.a.createElement(styles["MuiThemeProvider"], {
        theme: MUITheme
      }, /*#__PURE__*/react_default.a.createElement(StyledWrapper, null, /*#__PURE__*/react_default.a.createElement(CssBaseline_default.a, null), /*#__PURE__*/react_default.a.createElement(modules_Stream_JavaPerm, null), /*#__PURE__*/react_default.a.createElement(modules_DocTitle, {
        titleString: this.props.titleString
      }), /*#__PURE__*/react_default.a.createElement(StyledApp, null, /*#__PURE__*/react_default.a.createElement(StyledBody, null, /*#__PURE__*/react_default.a.createElement(StyledMainWrapper, null, /*#__PURE__*/react_default.a.createElement(modules_Main_Main, {
        errorMessage: this.props.errorMessage,
        infoMessage: this.props.infoMessage
      })))))));
    }
  }]);
  return App;
}(react["Component"]);
var App_mapStateToProps = function mapStateToProps(state) {
  return {
    theme: getTheme(state),
    titleString: '',
    errorMessage: getErrorMessage(state),
    infoMessage: getInfoMessage(state)
  };
};
/* harmony default export */ var modules_App_App = (Object(react_suber_lib["withBus"])(Object(react_redux_es["connect"])(App_mapStateToProps, null)(App_App)));
// CONCATENATED MODULE: ./index.jsx
function index_ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function index_objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? index_ownKeys(Object(source), !0).forEach(function (key) { index_defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : index_ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function index_defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
/*
 * Copyright (c) 2002-2017 "Neo4j, Inc,"
 * Network Engine for Objects in Lund AB [http://neotechnology.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */















// Create suber bus
var index_bus = Object(bus["createBus"])();
// Define Redux middlewares
var suberMiddleware = Object(bus["createReduxMiddleware"])(index_bus);
var epicMiddleware = Object(es["createEpicMiddleware"])(rootEpic);
var localStorageMiddleware = createReduxMiddleware();
var index_reducer = Object(redux_es["combineReducers"])(index_objectSpread({}, rootReducer));
var enhancer = Object(redux_es["compose"])(Object(redux_es["applyMiddleware"])(suberMiddleware, epicMiddleware, localStorageMiddleware), window.devToolsExtension ? window.devToolsExtension() : function (f) {
  return f;
});
var index_store = Object(redux_es["createStore"])(index_reducer, getAll(),
// rehydrate from local storage on app start
enhancer);

// Send everything from suber into Redux
index_bus.applyMiddleware(function (_, origin) {
  return function (channel, message, source) {
    // No loop-backs
    if (source === 'redux') return;
    // Send to Redux with the channel as the action type
    index_store.dispatch(index_objectSpread(index_objectSpread({}, message), {}, {
      type: channel
    }, origin));
  };
});

// Signal app upstart (for epics)
index_store.dispatch({
  type: APP_START
});
var index_renderApp = function renderApp() {
  var renderMethod =  false ? undefined : react_dom_default.a.hydrate;
  renderMethod( /*#__PURE__*/react_default.a.createElement(react_redux_es["Provider"], {
    store: index_store
  }, /*#__PURE__*/react_default.a.createElement(react_suber_lib["BusProvider"], {
    bus: index_bus
  }, /*#__PURE__*/react_default.a.createElement(modules_App_App, null))), document.getElementById('mount'));
};
index_renderApp();
if (false) {}

/***/ }),

/***/ "./modules/Guides/style.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js?!../../node_modules/postcss-loader/lib/index.js!./modules/Guides/style.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./modules/Help/html/help.html":
/***/ (function(module, exports) {

module.exports = "<article class=help> <section class=main> <div class=headings> <p class=title>Help</p> <p class=subtitle>What is all this?</p> </div> <div class=content> <p> Neo4j Browser is a command shell. Use the editor bar up above &uarr; to enter Cypher queries or client-side commands. Each command will produce a \"frame\" like this one in the result stream. </p> <p>Use the&nbsp;<code>:help</code> command to learn about other topics.</p> <p>New to Neo4j? Try one of the guides to learn the basics.</p> <table class=\"table-condensed table-help\"> <tr> <th>Usage:</th> <td><code>:help &lt;topic&gt;</code></td> </tr> <tr> <th>Topics:</th> <td><a help-topic=cypher>:help cypher</a>&nbsp;<a help-topic=commands>:help commands</a>&nbsp;<a help-topic=keys>:help keys</a></td> </tr> <tr> <th>Guides:</th> <td><a play-topic=intro>:play intro</a>&nbsp;<a play-topic=graphs>:play graphs</a>&nbsp;<a play-topic=cypher>:play cypher</a></td> </tr> <tr> <th>Examples:</th> <td><a play-topic=\"movie graph\">:play movie graph</a>&nbsp;<a play-topic=\"northwind graph\">:play northwind graph</a></td> </tr> </table> </div> </section> </article> ";

/***/ }),

/***/ "./modules/Help/html/images/neo4j-world.png":
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARgAAABkCAYAAABdGS+CAAAcC0lEQVR42u2dB3gVVRbH6UpRQcRVLIu6Luqu7q6oi2VdGwusZRULKqJrAwFRRFhYsACKBROaUgNxgQABQk0gJISSENJ77yG999dLzp47TPK9TGbem/fezLxJuOf7/p/guzPvvpl7ftx2zu3Thxo1atR6ggHAANS9qHtQ/ekToUaNmjtAGY76wGDW+WqMrdkWq8XcZmzR4p91+GcT/jfTaNb7YJn3UFfRJ0aNGjUxYBlstpqWmCxGTVZtouZg1jZYE7UI5ge/DB8efYbR5ydfg3XRS+Bwti/k1qVqjBZDq8Vq/hyvvYI+QWrUqAnB5Ra9WZedV5+m/ebcR51AcaTvIj6GosZsLV6bive4kT5JatSoceHyJ+yJNAXl+plnHJsgGi4dmnnsHxBacMAEl+wJ+kSpUaPWAZffGC36up0pq50GC1cJFeHtOLxqwHuO6PwCr/BhqPGol1CzUStQPqggVAKqHFWAikQdQK1HLUG9i5qE+iOqH31T1Kj1PLj01Zu18acLD5vchQvRjKMT4EJpiKFB1xbTzzv8EwRDCMqIAjdVjfJFTUHRSWVq1HqCWdstr9Zrq7Uzj00EKQBDNCtwMlRr6mFSQLogMEZtiIIxW2Ohr2uwMaFOoT5F3UHfIjVq6uy99MPeS9mmuOWSwaVDvkmrIK22sRtAbtgUDSeKGqDDchp08JBfkru9GzK0epy+UWrU1AWYu7WmNh0Z1kgNGDLpqzEZ4NatMZ0gILAJvdgIXKvWGmEk9mgkGEYFo/5C3yw1auoAzLTsuuRWqeHSoYzabPjXkYxOANy8JQaE7K3j2V1gMXjteejvHeEqaPahxtI3TI2aB81oMawLyvVrlwswIQWHYEV0cafj/+HXeEHAzAnLZ8qM9Y2Dc2XNYG0H0JmtsCW1Eq5aH+kKZCyobajR9E1To+aa3fHbMVejPkA5PwWhMbYmbIz7GuQCzPbEH+BkcWWn0w9YHQFlrYZucCEwIfAhILnYou/2+a6sGndXnx6hTYUaNafh8h5KgwJW6ajrnenBtC0Ne0c2wKw4NxOB0trF4Z/clwrNBnMXuCw4V8h89iYOk4Ts2l8uOAUWMrz6zcZoBlyjN0Wbh6w9/yFtMtSoiYbLIJTWBi4dWi36JlqTpmhV5GeyAWZ9zBKIr6rt5vyjN0fDzNA8+BzB8qcdCZ3/f/7ZQkHA3O0bLwiTgdgz+rt/CnjHl0BqbT3UajUIrnbQm01Qo20CrcnA/L3FoDNqjG0FBrN+E95yAo2dokZNEDCjeeBCdFD0TXQmjf++9I2yAeZI9q+wKaVUdK/jr7uTeOFSqzMxwytu+T/vTIR9ORXQajRAaUsVBOcHMIGZpFc29/gLXerycdBzsCTsbfC6sACO5+2x1mjK28xWs0FrajuJX0GHUNSodYfMKR7AvCD6BuhYc+PKz+rkAkxsRSy8dzLXqaHN2sTyLnAxWtq7rEQR3bk9Dg7kErDo4HD2DgTK2y7Vb1Hom7A/Y3M7glavM2lP49fdR5sVNWqdgHkYlcGCxYjyQ/VzBjCP1mmrZFumrtI0ML0MZydmJx9MhzUJ5bA86mKXodEVa87DxpQSZn9NUJ4/fHriRUnq+fHx5+Fwlq8Vh04GBM1+mt+GGrUuoHkQNcrpC9GRhpmtJoMcE73Lz85ghi6D1kRIsYEObtwUDfFV9ZBSnQQLQl6TBYjzgl+CxMrzeoNZV4TP5nbatKhRc9OMFoNXYkWEXmpnTaxMgC8iCyWBy4N+Sdgb0jBzLGSHsFw9ro5gzUNZ2y1khQ0h8xRtIdSouWEkrYLRYtSQyVGpnHRD7NdQp9PBsHWRbsPlcf8U7AnpmT01coKFq59jvmCX0a1zaSuhRs1V8wrvN/1EdnSLQevyZKmtlp35gJl8JcvG7sKFRFvXI6jk3AxoTyQtKPZkdCTZOW0o1Ki5BhiS/AkWnMuHGk0DM3fiqkN+Gz4b6rTNMOtUrttwGYq9n8z6JjiSvdMjcOmQX8radqNZX4WQuY42FmrUnIPLJFunnhGay/Q+yHDEmbSZZF5kR4o3M5SZzgladFVkGTquPApcSd8ptSIuHjfqzdpokt6CNhpq1MTBpT8qnevYj+1NgdzGZihsvAib41cIrtiQydCFIa/D1oSVcLG5HDLqGqXI68Loqf2pTE+ILB97Gi5EHwVOglpNhdbabnlDDa/ujt+O6YsaoJamhHUZjPodCYhDTUV9gnoXNRH1R9RwtTV/NojvLtQTqDfYOr+PehZ1P7uTlf6D4gZg3rIXxzP9RA7EVtaCzmyCBl0rJFYlQ1DubrILFpKqUqFR3wZakwmiKmpgamAW9PMOlwQuJGdMYnU9SJEfWEptiP0KDGZdBfZiBrrRqB9hG3GHHnRQfiBqCuor1HZUGCoPZUC1oxpR2eyOyy9QDxD4KOSgN6Fmo0LYTVjgQMmopaixHgJKP9R41LeoFBH1JapnN5e97mlI4vf/i33eHZqu0Du2ba+viIXLIFSh2KDBu3zjmUDE1Qkl8FN8CbwelAW/3x4nGVRsRXbtVrTWML0GNQGG9NjKmgs11nbrDDde2GZOA/5JoNwI1GJUuUhHsFUlaj7qSpka3f0s0NpdqFuHEkjPQSHHHIpagqp2o75EZtQR1H0egMvNKBOnPl4KfO9EznfmiwXMLKnBIJXIUIsMu9QElw55X1gARou+3tWjcx0BBv/eH7WKEyLvqipQ0yRu5DvdBAtXB1C/lck5SDTwxxKAhSsrajfqDgUBs4ZTB9JjHK1OwHiFD0FVOnJ0kpR73K5EGLLuvGJwITFGbUYdSJmAXOpeTJO+XoOAeVhqwLAOcVBE4yYOU+QEhFYTcLnZ0F5w8H3kX9cIFkAEkAtQy1E+qBOoFjvXkpQAz0vsGPegch08lwa2J7bLps7L2Hd0lM19Ag5+8wIFnHwkz7P3UQhsLgHmP/acnMT7bEur6gw21JqsMCM0TxHAzDtbABdKI1QJlw6R413MVtMPUgIG/zuEncvgNmId6jAZb7M9iAGc+5HrHkOtRGXacYZAVyGD181lwca9pwW1F/Uy6ioRvYmJ7DySSeBe70k4V9Eq8ByKWZj8TczzwDK3oGahTtrpufnJNRxl67CM5x+ZO9UJGK/wwahGe06+MqaEN2XC4xJsnHOk8LJq8En8TtWAWR31HyB5dCQGDLfnQhrzNmcCzNiVpal2/uVe5UJ9vXjuQ+q2j6zAuPgM7kQFCNRxoZsO8YUACKpYUAxw495/EfhHoGNOabQMDj6UnWjuMqxUcGjmNGCedeTkpTwpLYmR3LhywoUMxYwWCxNwqGbAzAqcBCaLkRyRO1oKwKCe5HGG8W4uGfsLOMLLTtxnpsCwYqJEjfc1tofGhddEF+/3qcBvJs94qIRO9ww7v8W3SjZUYgefx/M949QMmE2OHL1Bb+YFjJ97eXEdiqxK1etaVA2XDtVoyltJqgsJAOPNNsyOv5dK0f1lezPePI2zjAxXRFw/jl0K5zrQGIkb8EMsUG2/pxZ1o5P3+Qc7zLK9jx71pozLtwk8zzdAqm0C7BaFMs79Tym8euUEYLzC+6JKHTk6AYmYY0WkFhmC5TeW9AjA5NantuAjeUUCwNg2oFYpVybYfR98GclmOrhuODtXYXsNGXZdK+OyN7cnc8aJ68eimjjXk7//VWbnG8xOBnOf7wqJ7v8uz72fVjNg7hPj6CRRdkqtpgtcfNOrXD3iVbTIZr3EyuQeAZiYsjASADlXAsDY6kMZGsgoHucrtrdDFT/7gadHcbvMDfkNnufxvMhrL/BMgk5WyAGH8GzaI8O8+yX4xyGbc984D+y/cQowS8Q6O0kS9cLhDPj4dD6M352syArSJ2cKIPxiWI8ATHCef7u13fqdhIA5LWMj+VbsOJ5dEm3jlH1Oocbsx/neGJErRtzftlhhJxzDMxEb6uY9p/D8rilqB0yUWjfXXQq0zINIlS9Rdyis8JAZAfOlhID5h4yN5DqeIcgikTA6oWBjvo0n5OBpO+X78yzNn+njAWOX6iUbzpDeCude2UqFgLgGGK/wUah2NQOG9JjSarJ6BGCSKyPJGPIDiQCTI3fj4VleDeUpM4yzIc7q6lK0G/X8hVPPo3bKvsPzLP/ax0OG3x0vxZCGgInnd73nod8kGjCTnXH24T9fgHWJ5ZBWp4HwsmYm8FHuORgSiV3SUtUjAFPclEMmeZ+VCDCLFWgoizjfWS5iuHHaAw36IU4dyHBtoEDZ05yyx/p40HicEVwBNM/EvKiVP08D5j1n5l+Sa9q6rSItDC+UFTC3bIkBjUnPbMdXO2AadbXkAY2TCDCPKNBQHuPGsvCU2SD3pLOIevZlgzVt6/E4T7lreHYEP9rHw8bZcuD0xkF2ewC3fcxXETQFAbNUrKMLHd9KDqIfuDpCNsCQyOwGvQ6+OvO+quHy+clXwWK1kI12QyUADNlrcoUCDeV3PA33ak6ZfM5KyLUeatTbOPX8hqfMVE6ZOjXkbiFL1Jx6RTh5fQDPxsZhPQEwP4t19GVRFwWPb73NJ1bWXsyuzHIIyPRRNWB+TVoFWlNbhIsvjAuYeIUaynAewNzGWQmxO4RSsFEvEjFfxF1x2tlHBUbmgHhirEaKvHYsT8zXCg//HtGACRDr5K8ey+KFS5vJwnt8q5R65VgmZNbmqBowqVXRWmu7dY5EgDmuYGPhxujcafPZJM5n4R5s1G9x6pLKU4Yb6ayWTIP92N6U3SGewLXbeaLMr+spgIkU6+QEItGVLd0AMycsX/aVpKvXR4LRYob5wS+rEi6zg54lcUhGfBy3SgSYXQo2FqsdwLzN+SyM3X7vCS3mxmbx/JZ6T8XniHjOMZy6TRVxzU08S/TrVPBbRAOm0NnAw2+iS+BCRQsEFjZ0OxtaTvlnV8CJvP2qBAx7jnWUGy9ss6cakQPALJQ4KZOUstgu47MxOtze2M0qAgw3fOBTEdd48+SaubUnAUar5j0wtrpjWxxz9vTCkKmqgsvc4y+A3qzVY+/lQQkB85NKAPOTigHTZUKazdHC3Zo/UEWA2cKp3w8Oyl/Ls3t6h0p+i2jANPcUwBBtTS2F00XHVQWYo9k7Se8l2M0XplbAbFc5YAbZ1PU+7l6ZPioynliurQ7Kf8UDzHt6GmCy3c30ryRgRm+OZnox35ybpQq4kBMvdSa9BXsv9/RSwKznblpDfa4SzeeZrxAEkAqc0odTt+/slB3CM590VEW/RTRgzrri6Hf7xkPoxUYmbWZRs55JaSnHSQJ8ejc4B2q1jcy+E08PjcpaKuFQXvUhCV6YWgHzBeezH/uo1EhWOp7fcquK6hfIqdtcO2X5kmSN74mA2eOsg1+/MQrqdKZuq0n/CS9SrCezPrEEsutzYVbgZM8k+T42ARIr42FvdgX09w5/rhcDZgY3z2wfFRubQsK2vg+qqG7cQMVXBcoNZBOMybY9gA3CfMVGo+UCzGpnnXtRRBHvfhiS8U6pIRNZMj9TUgPnS8KY42mVPkEgOD8AkmsaYPBa5mSFcb0YMC9yPjuncsCkcur7tkrq1Z8n/85jAmX/zdN7mSRxfbirbS/KBZiFzjr3L8kVgjt6ycH0SvViRvxyAeKr6iGlOgk+PfGiInCZE/QsRJdFQE5DM9y0OaajLqN7MWD+xLNFvb+KAePLqa+/Sur1N57l5mt4ypGYqyxuSlIZ6qMYYKY769jvBOfwwiWrQav4qtKV2IPYmVkO5a3V8OXpd2WFy6LQN6CwsRiCCqvgqvWdICWpLgb0YsD05dmB+qiKAfMST4rMASqoF3cFKUyg3Is8vZfXezJgnnLWqcmRsWdLm7rAxWCxwpP7Uj2ydE2GZZ+dzYcWgw72pK1nsvtLCRYyBPNNWgWN+lb4NrqIO5ldJdELVyVg2M/3cj5fqWLADOVJSv60h+vUlyfN5acCZbm7fQvk6DEqCZihKIOzTk2ip2eH5cPurBpYnVDOnFHt6T0yD+9JhujKOqhqq4ftiT+4PTdD5lo2xS2D0pZKSKtthAkH0vi+d+9lABhukuk8NUQo2/k9J9SQzY4zoSoYUGpT7kmech/JVCeFAHMJMid60mY7R72Zfx5MZ86xLsNh07EcP1gZPodZ9RELleVnZ8Dh7F+hqKkUCptamCBPO0vw0y4DwNzIHvBuW+YtFQPmVU9n3bepSz+e9J3nBMqG8pyDdUVvAMwcKZybHC37fkgu06P56FSeomdX8+WQmRiQBptSSqGirY0Z3kSWnIOgXD9mGEV6Jj9FzocNsV+DX+o6CMzZBRElYVCrbUJpwTe9jImzchAlbkWN7O2AYctwd/Tmq2Fuw86QhJumMtYTk9MCR4yM5yl3v5IZDZUGzBh3HZpMenKPNMlr1MF1G6JU0au5938JMPd0PvwYV4LDujI4W1oF6bX1EFFWDXuzy+Cn+BL47GwBjNuV6MyGwUgJX7jaAXM7Ty9mkYp7MU9LcUSum3X4A08s0SGBsvs55Vr4Vpl6JmAuQSbdHSf+PraUd2Vps8zHynpYiy8XwLDl/scTzfyUAvXrz274G+DkdSFKrMgIfPcIdoKW+7zu4il7J887+F7m+ikOmO/dcTaSvkEtS9cK6o+XGWBu4YmPqZPy1Eme7ySnJB6xOX51oBPX3sJz/KxWqjO0HcDlHA/c5gmU9+E52vY3vQ0wj7rjbEfy63kBc66subfC5SJz5O5lBBi27ASe8nVCu1LdrBdJV3CeJ9jyCifuMZ5n2doiJheLi3Uey66yceHiI1B+NE9CqU0KvHPFAdMfVeCqw5ElXD4jKzBqW2V67lA6eCeUwdLzxTBmq8u5hFdK/MJ7BGDY8kt5HMjIDmOkOuCdpMes5vkekrTpSifvNV0g1QMZ8l0v4cTymzzhAEyCbztHrHjxwO/23geYS5CZ5o7zvnU8G6o0RgYs9ToTs5LkyNnJdv/bfWKZlJhKxDFtS6vqAkCNyeLKBsEm1IjLGDB9eZIndW5rdyduBq/9C9m3InBvX1dXgvC6mTzHmXScr7TMnQz9bDrPJIE6k7OMRtgZSnEngfcq9M49Aph+qDR3ewgjN0TxwuLmLTEwMzQPThZXQUVbK5isFjCjmg0asLa3g95sgtKWFjiQWwHTEFYjJV6BEuplFTbrnYXbf2V44T0GMDbXzee5tkMJbMKk+x31avDzu1knz7KTWOpHCX4niQeqEbg/ibHawUYWXyUCsOPYOifZqfNqe0DEz77kuSaGnWuSW6A8YC5B5nmphySTD6ZDYnU96MxGiKuIhR3J3rDszAcwL3hK52FqZNctye9CEkn5p2+A1Op0MFjMEHqxGh7YlSRJXZbbOXaFJLISeZ9qZvczBUzHtc9zjpUVct5UdiMZcWJ/dthQwE66Orp2hoS/lUz8Rjv4TjLcS7Op7yoyL4I6zAKgxsH1rY6iuNmEUnUqygyoGGD6oqKkcOhH9iRDVEUdVGvqwTfpR6dzt5CETvsztjA9nKP5lW6HI5DTD/jMgr2nK9eK3hQ4RyYn75GAYa8fRZKU80xWuiMjm/B6hAy/ty+72zdHYic1sL2W60TU4ROVpR5VCDCXIPOEu72WpecLGTBIEXxIQEO2/Lca9TA10PVJ41E45KrlSZS1KUX0Xp1i1CAKGMH73MZOnLa60dCbyWFpCk10kv0176My3HROMrG7lfSORH4vybxXcvkC5hJkQlxxYhIesD+nAi42l8OSU9MljWr+PmIuNOha4PvYYpfTc/55ZyLEVrV2Hnm7PqncmZCGt2Vs7Peyw40O3a0gYJ7jfPdQN+83iF3OXsf2EowOeioZ7OkFT3gq/ICF41x2c57OgTO2s7/Lmw1QdHYD4DWc5+0JtXsaMPeidM7GIkXjkCi+IpbpdciRk2VhyOudPY++IhJSkXiiKUczmd6L7WfD1kUyaSec+H1RzFI+NVcdeAQ7mUsc8jHU78mxtT2gvk+R3b8shMexicUH9IL34WHAXILMVGeGRTsyyiGlOln2FJYke11lWw0sDC8QrM/f/VO6DIeaDGZm/4uLw6ty1A0UE9R6EfBVAJhLkFkpxgnnnc1nssoplbpyadg70GLQwqSA7tAgaTsr2ozd5loIZFwIvtSjHqBNkhoFjDyAIXtjjtlzwrG+cdBm1MuespKrNVGLoFGvZzbp2dbnb9h7EbIXnT/q9g3aHKlRwMgFmEuQuRqVKeSEh/IqITB3j0eOD4mriIblUV2PSxm/O1kQMCQZlRNw+YE2RWoUMHID5hJkfodq5Drh/bsSmaHKJyf+5RHAfHXmfew9GboMfchkc36TrhtcKjXGbr0dOwqik7rUKGAEr5/CuT7N/Vp5hT+DMtk64sG8CjiUtd2jJyxGlpyBZZxeDFmOJgmvOqykxcBs/BMJlwzUNbQZUqOAEbx+Luf6s9LUzCv8EXa7PAxaEwEakxEWh07zKGDWRv8X0mrru4GC1I+EGZAhkxM7dYNRw2kTpEYBY/f67znXB0hXO6/wW1AJT+9PheKmco8fQD878J9MnJPNQWiuahUdFlGjgBF1/WF5d557hQ/em12ZeyLP3+OAIYqviIcPQnJdBQs5tuUt2uyoUcCIuvZ2Nm+NbEfbMtZmbAn3SVipCsAE5e6CVXElrsClAvUQbXLUKGBEX/srT4rPIZJXUmtszfO6sEAVgNmduh72ZJc5C5cY1I20uVGjgHEMGCxzAyqIL++NLJU0mPWNSm+uE9LGuK/hXGmVM72WGaiBtKlRo4DpChj88zQ26HM36mc2QVUa21PhwqURNVKWSposRu2i0DdVARiykhRXWesILA2oBWT+iDYxahQwgoBZKDLFA0kW9ohsldSZNOXfRXysCsCQw+mPFVQIgUWDWkH3tlCjJhlgTpLoclkrqTG2JvwS+6UqAHMw04c5JpYDljrUatT1tElRo9YFMBPYZOUdGiUCMCRPTiBKmUUR7MEc3Jv2iyoAc7ooCJacLyZQSWUjwB+me1qoUXMJPsPYBGjPoF5gE7dfp3hFAODDjJp4jRoAU6upNQQXNUyhzYMatV5iCJibzFaTieyk9SRcvjj9bzBZjBqsD+2xUKPWm0xrastbH7PUo4A5kLEFyHCNvg1q1HqZWazmT4oas7Ud5xt5Ig6pzdBMwqafom+DGrXeN0y6wmDW166NXuwRwOxL39SOvZdo+iaoUeulZm23/ruqtUQz89hEReFC8v/qTFo9Qo7GElGj1ot7Mf31Zm3yqYIAk1JwIScXZNYm6PB799A3QI1a74fMDUaLvp4cEasEYILz/M16sy4dv/dK+vSpUbs8IDPOZDHoSbZ/OeHil7K2HWHWSM6sp0+dGrXLyCxWy2SjxaD1T99olXpliczxnCk6YkS41CJc7qNPmxq1y7Mnc5fBrCuLLj1lkOq0gYUhUyGvPo3MuSTg/WlsETVqlzlkhutMmmMIBP2BjC3ts4OedQks84KnQGjBAZPJYjTozboNeN9B9OlSo0atAzQPIGgiNcZWXUj+fsuqyM/A0XI2gdH6mCUQXhxkNJj1Bp1JG4D3uYM+TWrUqAmB5mGjxbAWYVFGoFHQkNkSV35WezJ/n5Wcq3SqIMCSWHleU9yU00Lim7Smthz87wq87m769KhR61n2f4br28HwexuoAAAAAElFTkSuQmCC"

/***/ }),

/***/ "./modules/Help/html/intro.html":
/***/ (function(module, exports, __webpack_require__) {

module.exports = "<article class=help> <section class=main> <div class=headings> <div class=box-max> <img style=width:80% src=" + __webpack_require__("./modules/Help/html/images/neo4j-world.png") + " class=img-responsive> </div> </div> <div class=headings> <p class=title>Enterprise Data Import</p> <p class=subtitle>Easy Neo4j Integration, try by yourself</p> </div> <div class=content> <p> This is the official Neo4j ETL tool you use to retrieve and map the metadata from your relational database and many other different sources like Hadoop, Cassandra, MongoDB, Couchbase and so on. You can easily drive the export from these sources and import into Neo4j database. Also, with the newly released user interface you can preview the resulting graph data model and eventually improve it by setting your favourite label and property names or relationships and property types. It supports all relational databases with a JDBC driver, for example MySQL, PostgreSQL, Oracle and Microsoft SQL. </p> </div> </section> </article>";

/***/ }),

/***/ "./modules/Help/html/unknown.html":
/***/ (function(module, exports) {

module.exports = " <article class=\"help alert\"> <section class=main> <div class=headings> <p class=title>Unrecognized command</p> <p class=substitle>.</p> </div> <div class=content> <p>Apologies, but that was unparseable or otherwise unrecognized</p> <table class=\"table-condensed table-help\"> <tr> <th>Try:</th> <td> <ul> <li><a help-topic=help>:help</a>&nbsp;- for general help about using Neo4j Browser</li> <li><a help-topic=cypher>:help commands</a>&nbsp;- to see available commands</li> <li><a href=https://neo4j.com/docs/ >Neo4j Documentation</a>&nbsp;- for detailed information about Neo4j</li> </ul> </td> </tr> <tr> <th>Keys:</th> <td> <ul> <li><code>&lt; ctrl - &uarr; &gt;</code>&nbsp; to retrieve previous entry from history.</li> <li><code>&lt; ctrl - &uarr; &gt;</code>&nbsp; to retrieve previous entry from history.</li> </ul> </td> </tr> </table> </div> </section> </article> ";

/***/ }),

/***/ "./modules/Stream/JavaPerm.jpg":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/JavaPerm-cdda22625231f23412673a5b4f85dd05.jpg";

/***/ }),

/***/ "./styles/font-awesome.min.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js!./styles/font-awesome.min.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./styles/fonts/Inconsolata-Bold.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/Inconsolata-Bold.ttf";

/***/ }),

/***/ "./styles/fonts/Inconsolata-Regular.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/Inconsolata-Regular.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-Bold.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-Bold.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-BoldItalic.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-BoldItalic.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-ExtraBold.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-ExtraBold.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-ExtraBoldItalic.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-ExtraBoldItalic.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-Italic.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-Italic.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-Light.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-Light.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-LightItalic.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-LightItalic.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-Regular.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-Regular.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-Semibold.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-Semibold.ttf";

/***/ }),

/***/ "./styles/fonts/OpenSans-SemiboldItalic.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/OpenSans-SemiboldItalic.ttf";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.eot":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.eot";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.eot?v=4.7.0":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.eot";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.svg?v=4.7.0":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.svg";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.ttf?v=4.7.0":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.ttf";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.woff2?v=4.7.0":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.woff2";

/***/ }),

/***/ "./styles/fonts/fontawesome-webfont.woff?v=4.7.0":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/fontawesome-webfont.woff";

/***/ }),

/***/ "./styles/fonts/neo4j-world.eot?9li18m":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/neo4j-world.eot";

/***/ }),

/***/ "./styles/fonts/neo4j-world.svg?9li18m":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/neo4j-world.svg";

/***/ }),

/***/ "./styles/fonts/neo4j-world.ttf?9li18m":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/neo4j-world.ttf";

/***/ }),

/***/ "./styles/fonts/neo4j-world.woff?9li18m":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/neo4j-world.woff";

/***/ }),

/***/ "./styles/fonts/streamline.eot":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/streamline.eot";

/***/ }),

/***/ "./styles/fonts/streamline.svg":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/streamline.svg";

/***/ }),

/***/ "./styles/fonts/streamline.ttf":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/streamline.ttf";

/***/ }),

/***/ "./styles/fonts/streamline.woff":
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "assets/fonts/streamline.woff";

/***/ }),

/***/ "./styles/inconsolata.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js!./styles/inconsolata.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./styles/neo4j-world.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js!./styles/neo4j-world.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./styles/open-sans.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js!./styles/open-sans.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ "./styles/streamline.css":
/***/ (function(module, exports, __webpack_require__) {

// style-loader: Adds some css to the DOM by adding a <style> tag

// load the styles
var content = __webpack_require__("../../node_modules/css-loader/index.js!./styles/streamline.css");
if(typeof content === 'string') content = [[module.i, content, '']];
// Prepare cssTransformation
var transform;

var options = {}
options.transform = transform
// add the styles to the DOM
var update = __webpack_require__("../../node_modules/style-loader/lib/addStyles.js")(content, options);
if(content.locals) module.exports = content.locals;
// Hot Module Replacement
if(false) {}

/***/ }),

/***/ 0:
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__("./index.jsx");


/***/ })

/******/ });