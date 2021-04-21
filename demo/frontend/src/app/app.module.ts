import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { iconsPathFactory, TuiDialogModule, TuiLinkModule, TuiNotificationsModule, TuiRootModule, TUI_ICONS_PATH } from '@taiga-ui/core';
import { environment } from 'src/environments/environment';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AskerModule } from './components/asker/asker.module';
import { TOKENAPI } from './tokens/token-api';
import { TuiTagModule } from '@taiga-ui/kit';
import { HttpLoaderInterceptor } from './shared/interceptors/http-loader-interceptor';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    TuiRootModule,
    HttpClientModule,
    AskerModule,
    TuiTagModule,
    TuiDialogModule,
    TuiNotificationsModule,
    TuiLinkModule
  ],
  providers: [
    { provide: TOKENAPI, useValue: environment.backendUrl },
    {
        provide: TUI_ICONS_PATH,
        useValue: iconsPathFactory('assets/taiga-ui/icons/'),
    },
    {
			provide: HTTP_INTERCEPTORS,
			useClass: HttpLoaderInterceptor,
			multi: true
		}
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
