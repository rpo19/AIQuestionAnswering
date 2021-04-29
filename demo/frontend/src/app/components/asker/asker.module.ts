import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AskerComponent } from './asker.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { TuiInputModule, TuiTagModule, TuiTextAreaModule } from '@taiga-ui/kit';
import { TuiDataListModule, TuiDropdownControllerModule, TuiDropdownModule, TuiGroupModule, TuiLinkModule, TuiLoaderModule, TuiSvgModule, TuiTextfieldControllerModule } from '@taiga-ui/core';
import { TuiButtonModule } from '@taiga-ui/core';
import { TuiHintModule } from '@taiga-ui/core';
import { TuiMapperPipeModule } from '@taiga-ui/cdk';
import { PolymorpheusModule } from '@tinkoff/ng-polymorpheus';
import { CodeMarkupModule } from 'src/app/shared/components/code-markup/code-markup.module';
import { GraphModule } from 'src/app/shared/components/graph/graph.module';
import { TuiHostedDropdownModule } from '@taiga-ui/core';
import { TuiRadioBlockModule } from '@taiga-ui/kit';
import { TypeOfPipeModule } from 'src/app/shared/pipes/type-of.pipe';
import { TuiDataListWrapperModule, TuiSelectModule} from '@taiga-ui/kit';



@NgModule({
  imports: [
    CommonModule,
    TuiInputModule,
    ReactiveFormsModule,
    FormsModule,
    CodeMarkupModule,
    GraphModule,
    PolymorpheusModule,
    TuiTextfieldControllerModule,
    TuiButtonModule,
    TuiLoaderModule,
    TuiTagModule,
    TuiHintModule,
    TuiSvgModule,
    TuiMapperPipeModule,
    TuiDropdownModule,
    TuiHostedDropdownModule,
    TuiDataListModule,
    TuiDropdownControllerModule,
    TuiLinkModule,
    TuiRadioBlockModule,
    TuiGroupModule,
    TypeOfPipeModule,
    TuiDataListWrapperModule,
    TuiSelectModule,
    TuiTextAreaModule
  ],
  declarations: [AskerComponent],
  exports: [AskerComponent]
})
export class AskerModule { }
