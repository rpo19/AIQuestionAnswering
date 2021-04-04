import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AskerComponent } from './asker.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { TuiInputModule, TuiTagModule } from '@taiga-ui/kit';
import { TuiLoaderModule, TuiTextfieldControllerModule } from '@taiga-ui/core';
import { TuiButtonModule } from '@taiga-ui/core';


@NgModule({
  imports: [
    CommonModule,
    TuiInputModule,
    ReactiveFormsModule,
    FormsModule,
    TuiTextfieldControllerModule,
    TuiButtonModule,
    TuiLoaderModule,
    TuiTagModule
  ],
  declarations: [AskerComponent],
  exports: [AskerComponent]
})
export class AskerModule { }
